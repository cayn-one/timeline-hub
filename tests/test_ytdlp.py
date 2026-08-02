import asyncio
import contextlib
import re
import stat
import urllib.error
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from timeline_hub.infra import ytdlp as ytdlp_module
from timeline_hub.infra.ytdlp import YtDlpAuthenticationError, YtDlpCookieFileError, YtDlpMetadataError

_COOKIE_FILE = Path('/tmp/timeline-hub-test/youtube-cookies.txt')
_COOKIE_DATA = b'# Netscape HTTP Cookie File\n.example.com\tTRUE\t/\tTRUE\t0\tSID\ttest-value\n'


def _assert_yt_dlp_common_flags(args: tuple[str, ...]) -> None:
    assert '--ignore-config' in args
    assert '--quiet' not in args
    assert '--no-warnings' not in args


def _assert_cookie_file_argument(args: tuple[str, ...], cookie_file: Path) -> None:
    assert args[args.index('--cookies') + 1] == str(cookie_file)


def _assert_ejs_arguments(args: tuple[str, ...]) -> None:
    assert args[args.index('--remote-components') + 1] == 'ejs:github'


def _assert_no_failed_experiment_arguments(args: tuple[str, ...]) -> None:
    assert '--impersonate' not in args
    assert '--extractor-args' not in args


@pytest.fixture(autouse=True)
def _supply_cookie_file_to_legacy_public_calls(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    _COOKIE_FILE.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    _COOKIE_FILE.write_bytes(_COOKIE_DATA)
    fetch_track_info = ytdlp_module.fetch_track_info
    download_audio_as_opus = ytdlp_module.download_audio_as_opus
    get_media_duration = ytdlp_module.get_media_duration
    download_audio_as_opus_clipped = ytdlp_module._download_audio_as_opus_clipped

    async def _fetch_track_info(url: str, **kwargs: object) -> ytdlp_module.UrlTrackInfo:
        return await fetch_track_info(url, cookie_file=kwargs.pop('cookie_file', _COOKIE_FILE), **kwargs)

    async def _download_audio_as_opus(url: str, **kwargs: object) -> ytdlp_module.DownloadedAudio:
        return await download_audio_as_opus(url, cookie_file=kwargs.pop('cookie_file', _COOKIE_FILE), **kwargs)

    async def _get_media_duration(url: str, **kwargs: object) -> timedelta | None:
        return await get_media_duration(url, cookie_file=kwargs.pop('cookie_file', _COOKIE_FILE), **kwargs)

    async def _download_audio_as_opus_clipped(url: str, **kwargs: object) -> bytes:
        return await download_audio_as_opus_clipped(url, cookie_file=kwargs.pop('cookie_file', _COOKIE_FILE), **kwargs)

    monkeypatch.setattr(ytdlp_module, 'fetch_track_info', _fetch_track_info)
    monkeypatch.setattr(ytdlp_module, 'download_audio_as_opus', _download_audio_as_opus)
    monkeypatch.setattr(ytdlp_module, 'get_media_duration', _get_media_duration)
    monkeypatch.setattr(ytdlp_module, '_download_audio_as_opus_clipped', _download_audio_as_opus_clipped)

    original_setattr = monkeypatch.setattr
    cookie_aware_helpers = {
        '_download_audio_as_opus_clipped',
        '_download_audio_as_opus_internal',
        '_download_cover_as_jpg',
        '_download_track_metadata',
        '_get_media_duration',
        '_get_selected_audio_codec',
        'get_media_duration',
    }

    def _setattr(target, name, value=..., raising: bool = True):
        if target is ytdlp_module and name in cookie_aware_helpers and callable(value):
            original_value = value

            async def _without_cookie_file(*args: object, **kwargs: object) -> object:
                kwargs.pop('cookie_file', None)
                return await original_value(*args, **kwargs)

            value = _without_cookie_file
        return original_setattr(target, name, value, raising=raising)

    object.__setattr__(monkeypatch, 'setattr', _setattr)
    yield
    _COOKIE_FILE.unlink(missing_ok=True)
    with contextlib.suppress(OSError):
        _COOKIE_FILE.parent.rmdir()


@pytest.mark.asyncio
async def test_cookie_file_is_added_to_every_ytdlp_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, tuple[str, ...]] = {}

    async def _fake_run(
        *,
        operation: str,
        args: tuple[str, ...],
        timeout: timedelta,
    ) -> tuple[bytes, bytes]:
        observed[operation] = args
        if operation == 'duration_probe':
            return b'10\n', b''
        if operation == 'codec_probe':
            return b'opus\n', b''
        output_template = Path(args[args.index('-o') + 1])
        if operation == 'audio_download':
            output_template.with_suffix('.opus').write_bytes(b'OggS-audio')
            output_template.with_suffix('.jpg').write_bytes(b'cover')
            output_template.with_suffix('.info.json').write_text('{"artist": "Artist", "title": "Song"}')
        elif operation == 'cover_download':
            output_template.with_suffix('.jpg').write_bytes(b'cover')
        elif operation == 'metadata_download':
            output_template.with_suffix('.info.json').write_text('{"artist": "Artist", "title": "Song"}')
        return b'', b''

    monkeypatch.setattr(ytdlp_module, '_run_yt_dlp_command', _fake_run)

    assert await ytdlp_module.get_media_duration(
        'https://example.com/watch?v=abc',
        cookie_file=_COOKIE_FILE,
    ) == timedelta(seconds=10)
    assert (
        await ytdlp_module._get_selected_audio_codec(
            'https://example.com/watch?v=abc',
            cookie_file=_COOKIE_FILE,
        )
        == 'opus'
    )
    await ytdlp_module._download_audio_as_opus_internal(
        'https://example.com/watch?v=abc',
        cookie_file=_COOKIE_FILE,
        download_cover=True,
        with_metadata=True,
        timeout=timedelta(seconds=5),
    )
    await ytdlp_module._download_cover_as_jpg(
        'https://example.com/watch?v=abc',
        cookie_file=_COOKIE_FILE,
        timeout=timedelta(seconds=5),
    )
    await ytdlp_module._download_track_metadata(
        'https://example.com/watch?v=abc',
        cookie_file=_COOKIE_FILE,
        timeout=timedelta(seconds=5),
    )

    assert set(observed) == {
        'duration_probe',
        'codec_probe',
        'audio_download',
        'cover_download',
        'metadata_download',
    }
    for operation, args in observed.items():
        cookie_argument = Path(args[args.index('--cookies') + 1])
        if operation == 'duration_probe':
            assert cookie_argument.parent == _COOKIE_FILE.parent
            assert cookie_argument != _COOKIE_FILE
            assert not cookie_argument.exists()
        else:
            assert cookie_argument == _COOKIE_FILE
        _assert_ejs_arguments(args)
        _assert_no_failed_experiment_arguments(args)


@pytest.mark.asyncio
async def test_public_operation_uses_one_private_cookie_copy_and_preserves_canonical_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical_cookie_file = tmp_path / 'youtube-cookies.txt'
    canonical_cookie_file.write_bytes(_COOKIE_DATA)
    operation_cookie_files: list[Path] = []
    mutated_cookie_data = _COOKIE_DATA.replace(b'test-value', b'updated-value')

    async def _fake_run(
        *,
        operation: str,
        args: tuple[str, ...],
        timeout: timedelta,
    ) -> tuple[bytes, bytes]:
        operation_cookie_file = Path(args[args.index('--cookies') + 1])
        operation_cookie_files.append(operation_cookie_file)
        assert operation_cookie_file != canonical_cookie_file
        assert stat.S_IMODE(operation_cookie_file.stat().st_mode) == 0o600
        output_template = Path(args[args.index('-o') + 1])
        if operation == 'cover_download':
            assert operation_cookie_file.read_bytes() == _COOKIE_DATA
            operation_cookie_file.write_bytes(mutated_cookie_data)
            output_template.with_suffix('.jpg').write_bytes(b'cover')
        elif operation == 'metadata_download':
            assert operation_cookie_file.read_bytes() == mutated_cookie_data
            output_template.with_suffix('.info.json').write_text('{"artist": "Artist", "title": "Song"}')
        return b'', b''

    monkeypatch.setattr(ytdlp_module, '_run_yt_dlp_command', _fake_run)

    result = await ytdlp_module.fetch_track_info(
        'https://example.com/watch?v=abc',
        cookie_file=canonical_cookie_file,
        with_cover=True,
        with_metadata=True,
    )

    assert result.cover == b'cover'
    assert result.metadata == ytdlp_module.TrackMetadata(artists=('Artist',), title='Song')
    assert len(operation_cookie_files) == 2
    assert operation_cookie_files[0] == operation_cookie_files[1]
    assert not operation_cookie_files[0].exists()
    assert canonical_cookie_file.read_bytes() == _COOKIE_DATA


@pytest.mark.asyncio
async def test_public_operation_removes_private_cookie_copy_on_cancellation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical_cookie_file = tmp_path / 'youtube-cookies.txt'
    canonical_cookie_file.write_bytes(_COOKIE_DATA)
    operation_cookie_file: Path | None = None

    async def _cancel_download(
        url: str,
        *,
        cookie_file: Path,
        with_cover: bool,
        with_metadata: bool,
        max_duration: timedelta | None,
        timeout: timedelta,
    ) -> ytdlp_module.DownloadedAudio:
        nonlocal operation_cookie_file
        operation_cookie_file = cookie_file
        raise asyncio.CancelledError

    monkeypatch.setattr(ytdlp_module, '_download_audio', _cancel_download)

    with pytest.raises(asyncio.CancelledError):
        await ytdlp_module.download_audio_as_opus(
            'https://example.com/watch?v=abc',
            cookie_file=canonical_cookie_file,
        )

    assert operation_cookie_file is not None
    assert not operation_cookie_file.exists()
    assert canonical_cookie_file.read_bytes() == _COOKIE_DATA


@pytest.mark.asyncio
@pytest.mark.parametrize('stage', ['open', 'create', 'copy', 'chmod'])
async def test_operation_cookie_preparation_failures_are_typed_and_preserve_the_cause(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    canonical_cookie_file = tmp_path / 'youtube-cookies.txt'
    canonical_cookie_file.write_bytes(_COOKIE_DATA)
    failure = OSError(f'{stage} failed')
    original_cookie_open = Path.open

    if stage == 'open':

        def _fail_canonical_open(path: Path, *args: object, **kwargs: object) -> object:
            if path == canonical_cookie_file:
                raise failure
            return original_cookie_open(path, *args, **kwargs)

        monkeypatch.setattr(Path, 'open', _fail_canonical_open)
    elif stage == 'create':

        def _fail_temporary_file(*args: object, **kwargs: object) -> object:
            raise failure

        monkeypatch.setattr(
            ytdlp_module.tempfile,
            'NamedTemporaryFile',
            _fail_temporary_file,
        )
    elif stage == 'copy':

        def _fail_copy(source: object, destination: object) -> None:
            raise failure

        monkeypatch.setattr(
            ytdlp_module.shutil,
            'copyfileobj',
            _fail_copy,
        )
    else:
        original_chmod = ytdlp_module.os.chmod

        def _fail_operation_cookie_chmod(path: Path, mode: int) -> None:
            if Path(path).name.startswith('.youtube-cookies.txt.operation.'):
                raise failure
            original_chmod(path, mode)

        monkeypatch.setattr(ytdlp_module.os, 'chmod', _fail_operation_cookie_chmod)

    with pytest.raises(YtDlpCookieFileError) as exc_info:
        await ytdlp_module.get_media_duration(
            'https://example.com/watch?v=abc',
            cookie_file=canonical_cookie_file,
        )

    assert exc_info.value.__cause__ is failure
    with original_cookie_open(canonical_cookie_file, 'rb') as source_file:
        assert source_file.read() == _COOKIE_DATA
    assert not list(tmp_path.glob('.youtube-cookies.txt.operation.*'))


@pytest.mark.asyncio
@pytest.mark.parametrize('outcome', ['success', 'error', 'cancelled'])
async def test_operation_cookie_unlink_failure_warns_without_masking_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
) -> None:
    canonical_cookie_file = tmp_path / 'youtube-cookies.txt'
    canonical_cookie_file.write_bytes(_COOKIE_DATA)
    unlink_failure = OSError('unlink failed')
    original_unlink = Path.unlink

    def _fail_operation_cookie_unlink(path: Path, *args: object, **kwargs: object) -> None:
        if path.name.startswith('.youtube-cookies.txt.operation.'):
            raise unlink_failure
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'unlink', _fail_operation_cookie_unlink)

    if outcome == 'success':

        async def _success_run(**kwargs: object) -> tuple[bytes, bytes]:
            return b'12\n', b''

        monkeypatch.setattr(ytdlp_module, '_run_yt_dlp_command', _success_run)
        expected_exception: type[BaseException] | None = None
    elif outcome == 'error':

        async def _error_run(**kwargs: object) -> tuple[bytes, bytes]:
            raise RuntimeError('download failed')

        monkeypatch.setattr(ytdlp_module, '_run_yt_dlp_command', _error_run)
        expected_exception = RuntimeError
    else:
        started = asyncio.Event()

        async def _cancelled_run(**kwargs: object) -> tuple[bytes, bytes]:
            started.set()
            await asyncio.Event().wait()
            return b'', b''

        monkeypatch.setattr(ytdlp_module, '_run_yt_dlp_command', _cancelled_run)
        expected_exception = asyncio.CancelledError

    with patch.object(ytdlp_module.logger, 'warning') as warning_mock:
        if expected_exception is None:
            result = await ytdlp_module.get_media_duration(
                'https://example.com/watch?v=abc',
                cookie_file=canonical_cookie_file,
            )
            assert result == timedelta(seconds=12)
        elif outcome == 'cancelled':
            operation = asyncio.create_task(
                ytdlp_module.get_media_duration(
                    'https://example.com/watch?v=abc',
                    cookie_file=canonical_cookie_file,
                ),
            )
            await started.wait()
            operation.cancel()
            with pytest.raises(expected_exception):
                await operation
        else:
            with pytest.raises(expected_exception, match='download failed'):
                await ytdlp_module.get_media_duration(
                    'https://example.com/watch?v=abc',
                    cookie_file=canonical_cookie_file,
                )

    warning_mock.assert_called_once_with('failed to remove isolated yt-dlp cookie file')


def test_cookie_file_path_is_redacted_from_logged_command() -> None:
    command = ytdlp_module._with_common_yt_dlp_args(
        ('yt-dlp', 'https://example.com/watch?v=abc'),
        cookie_file=_COOKIE_FILE,
    )
    message = ytdlp_module._format_yt_dlp_failure(
        operation='duration_probe',
        returncode=1,
        args=command,
        stderr=b'failure',
    )

    assert '--cookies' in message
    assert '<redacted>' in message
    assert str(_COOKIE_FILE) not in message
    assert '--remote-components' in message
    assert 'ejs:github' in message
    _assert_no_failed_experiment_arguments(command)


@pytest.mark.asyncio
async def test_public_ytdlp_apis_require_absolute_cookie_paths() -> None:
    with pytest.raises(ValueError, match='cookie_file must be an absolute path'):
        await ytdlp_module.fetch_track_info(
            'https://example.com/watch?v=abc',
            cookie_file=Path('cookies.txt'),
        )
    with pytest.raises(ValueError, match='cookie_file must be an absolute path'):
        await ytdlp_module.download_audio_as_opus(
            'https://example.com/watch?v=abc',
            cookie_file=Path('cookies.txt'),
        )


@pytest.mark.asyncio
async def test_fetch_track_info_with_cover_and_metadata_returns_all_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {'calls': []}

    async def _fake_cover(url: str, *, timeout: timedelta) -> bytes:
        observed['cover_url'] = url
        observed['cover_timeout'] = timeout
        return b'cover-jpg'

    async def _fake_metadata(
        url: str,
        *,
        timeout: timedelta,
    ) -> ytdlp_module.TrackMetadata:
        observed['metadata_url'] = url
        observed['metadata_timeout'] = timeout
        return ytdlp_module.TrackMetadata(artists=('Artist 1', 'Artist 2'), title='Song')

    monkeypatch.setattr(ytdlp_module, '_download_cover_as_jpg', _fake_cover)
    monkeypatch.setattr(ytdlp_module, '_download_track_metadata', _fake_metadata)

    result = await ytdlp_module.fetch_track_info(
        ' https://example.com/watch?v=abc ',
        with_cover=True,
        with_metadata=True,
        timeout=timedelta(seconds=11),
    )

    assert result == ytdlp_module.UrlTrackInfo(
        cover=b'cover-jpg',
        metadata=ytdlp_module.TrackMetadata(artists=('Artist 1', 'Artist 2'), title='Song'),
    )
    assert observed['cover_url'] == 'https://example.com/watch?v=abc'
    assert observed['metadata_url'] == 'https://example.com/watch?v=abc'
    assert observed['cover_timeout'] == timedelta(seconds=11)
    assert observed['metadata_timeout'] == timedelta(seconds=11)


@pytest.mark.asyncio
async def test_fetch_track_info_with_cover_only_fetches_only_cover(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = {'metadata_called': False}

    async def _fake_cover(url: str, *, timeout: timedelta) -> bytes:
        return b'cover-only'

    async def _fake_metadata(url: str, *, timeout: timedelta) -> ytdlp_module.TrackMetadata:
        observed['metadata_called'] = True
        return ytdlp_module.TrackMetadata(artists=('Artist 1',), title='Song')

    monkeypatch.setattr(ytdlp_module, '_download_cover_as_jpg', _fake_cover)
    monkeypatch.setattr(ytdlp_module, '_download_track_metadata', _fake_metadata)

    result = await ytdlp_module.fetch_track_info(
        'https://example.com/watch?v=abc',
        with_cover=True,
        with_metadata=False,
    )

    assert result == ytdlp_module.UrlTrackInfo(cover=b'cover-only', metadata=None)
    assert observed['metadata_called'] is False


@pytest.mark.asyncio
async def test_fetch_track_info_with_youtube_cover_only_uses_direct_thumbnail_and_skips_ytdlp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    async def _fake_download_http_bytes(url: str, *, timeout_seconds: float) -> bytes:
        observed['thumbnail_url'] = url
        observed['timeout_seconds'] = timeout_seconds
        return b'\xff\xd8\xff\xe0jpg'

    async def _unexpected_create_subprocess_exec(*args: str, **kwargs: object) -> object:
        raise AssertionError('yt-dlp subprocess should not be spawned for youtube cover-only fetch_track_info')

    monkeypatch.setattr(ytdlp_module, '_download_http_bytes', _fake_download_http_bytes)
    monkeypatch.setattr(ytdlp_module.asyncio, 'create_subprocess_exec', _unexpected_create_subprocess_exec)

    result = await ytdlp_module.fetch_track_info(
        'https://www.youtube.com/watch?v=EuVFrXWIGAw',
        with_cover=True,
        with_metadata=False,
        timeout=timedelta(seconds=9),
    )

    assert result == ytdlp_module.UrlTrackInfo(cover=b'\xff\xd8\xff\xe0jpg', metadata=None)
    assert observed['thumbnail_url'] == 'https://i.ytimg.com/vi/EuVFrXWIGAw/maxresdefault.jpg'
    assert observed['timeout_seconds'] == pytest.approx(9.0, rel=0, abs=0.02)


@pytest.mark.asyncio
async def test_fetch_track_info_with_music_youtube_cover_only_uses_direct_thumbnail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    async def _fake_download_http_bytes(url: str, *, timeout_seconds: float) -> bytes:
        observed['thumbnail_url'] = url
        return b'\xff\xd8jpg'

    async def _unexpected_create_subprocess_exec(*args: str, **kwargs: object) -> object:
        raise AssertionError('yt-dlp subprocess should not be spawned for youtube cover-only fetch_track_info')

    monkeypatch.setattr(ytdlp_module, '_download_http_bytes', _fake_download_http_bytes)
    monkeypatch.setattr(ytdlp_module.asyncio, 'create_subprocess_exec', _unexpected_create_subprocess_exec)

    result = await ytdlp_module.fetch_track_info(
        'https://music.youtube.com/watch?v=EuVFrXWIGAw',
        with_cover=True,
        with_metadata=False,
    )

    assert result == ytdlp_module.UrlTrackInfo(cover=b'\xff\xd8jpg', metadata=None)
    assert observed['thumbnail_url'] == 'https://i.ytimg.com/vi/EuVFrXWIGAw/maxresdefault.jpg'


@pytest.mark.asyncio
async def test_fetch_track_info_with_youtu_be_cover_only_uses_direct_thumbnail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    async def _fake_download_http_bytes(url: str, *, timeout_seconds: float) -> bytes:
        observed['thumbnail_url'] = url
        return b'\xff\xd8jpg'

    async def _unexpected_create_subprocess_exec(*args: str, **kwargs: object) -> object:
        raise AssertionError('yt-dlp subprocess should not be spawned for youtube cover-only fetch_track_info')

    monkeypatch.setattr(ytdlp_module, '_download_http_bytes', _fake_download_http_bytes)
    monkeypatch.setattr(ytdlp_module.asyncio, 'create_subprocess_exec', _unexpected_create_subprocess_exec)

    result = await ytdlp_module.fetch_track_info(
        'https://youtu.be/EuVFrXWIGAw',
        with_cover=True,
        with_metadata=False,
    )

    assert result == ytdlp_module.UrlTrackInfo(cover=b'\xff\xd8jpg', metadata=None)
    assert observed['thumbnail_url'] == 'https://i.ytimg.com/vi/EuVFrXWIGAw/maxresdefault.jpg'


@pytest.mark.asyncio
async def test_fetch_track_info_with_youtube_cover_only_falls_back_across_thumbnail_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_urls: list[str] = []

    async def _fake_download_http_bytes(url: str, *, timeout_seconds: float) -> bytes:
        del timeout_seconds
        observed_urls.append(url)
        if url.endswith('/maxresdefault.jpg'):
            return b'not-jpeg'
        if url.endswith('/sddefault.jpg'):
            return b'\xff\xd8good'
        return b'\xff\xd8unused'

    async def _unexpected_create_subprocess_exec(*args: str, **kwargs: object) -> object:
        raise AssertionError('yt-dlp subprocess should not be spawned for youtube cover-only fetch_track_info')

    monkeypatch.setattr(ytdlp_module, '_download_http_bytes', _fake_download_http_bytes)
    monkeypatch.setattr(ytdlp_module.asyncio, 'create_subprocess_exec', _unexpected_create_subprocess_exec)

    result = await ytdlp_module.fetch_track_info(
        'https://www.youtube.com/watch?v=EuVFrXWIGAw',
        with_cover=True,
        with_metadata=False,
    )

    assert result == ytdlp_module.UrlTrackInfo(cover=b'\xff\xd8good', metadata=None)
    assert observed_urls == [
        'https://i.ytimg.com/vi/EuVFrXWIGAw/maxresdefault.jpg',
        'https://i.ytimg.com/vi/EuVFrXWIGAw/sddefault.jpg',
    ]


@pytest.mark.asyncio
async def test_fetch_track_info_with_youtube_cover_only_shrinks_timeout_budget_across_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_timeouts: list[float] = []

    class _FakeLoop:
        def __init__(self) -> None:
            self._times = iter((100.0, 100.0, 103.0, 106.0))

        def time(self) -> float:
            return next(self._times)

    async def _fake_download_http_bytes(url: str, *, timeout_seconds: float) -> bytes:
        observed_timeouts.append(timeout_seconds)
        if url.endswith('/maxresdefault.jpg'):
            raise urllib.error.URLError('first attempt failed')
        return b'\xff\xd8ok'

    async def _unexpected_create_subprocess_exec(*args: str, **kwargs: object) -> object:
        raise AssertionError('yt-dlp subprocess should not be spawned for youtube cover-only fetch_track_info')

    monkeypatch.setattr(ytdlp_module.asyncio, 'get_running_loop', lambda: _FakeLoop())
    monkeypatch.setattr(ytdlp_module, '_download_http_bytes', _fake_download_http_bytes)
    monkeypatch.setattr(ytdlp_module.asyncio, 'create_subprocess_exec', _unexpected_create_subprocess_exec)

    result = await ytdlp_module.fetch_track_info(
        'https://www.youtube.com/watch?v=EuVFrXWIGAw',
        with_cover=True,
        with_metadata=False,
        timeout=timedelta(seconds=10),
    )

    assert result == ytdlp_module.UrlTrackInfo(cover=b'\xff\xd8ok', metadata=None)
    assert observed_timeouts == [10.0, 7.0]


@pytest.mark.asyncio
async def test_fetch_track_info_with_non_youtube_cover_only_keeps_ytdlp_cover_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = {'cover_called': False}

    async def _fake_cover(url: str, *, timeout: timedelta) -> bytes:
        observed['cover_called'] = True
        assert url == 'https://example.com/watch?v=abc'
        return b'cover-jpg'

    async def _unexpected_download_http_bytes(url: str, *, timeout_seconds: float) -> bytes:
        raise AssertionError(f'unexpected direct thumbnail fetch for non-youtube url: {url}')

    monkeypatch.setattr(ytdlp_module, '_download_cover_as_jpg', _fake_cover)
    monkeypatch.setattr(ytdlp_module, '_download_http_bytes', _unexpected_download_http_bytes)

    result = await ytdlp_module.fetch_track_info(
        'https://example.com/watch?v=abc',
        with_cover=True,
        with_metadata=False,
    )

    assert result == ytdlp_module.UrlTrackInfo(cover=b'cover-jpg', metadata=None)
    assert observed['cover_called'] is True


@pytest.mark.asyncio
async def test_fetch_track_info_with_metadata_only_fetches_only_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = {'cover_called': False}

    async def _fake_cover(url: str, *, timeout: timedelta) -> bytes:
        observed['cover_called'] = True
        return b'cover'

    async def _fake_metadata(url: str, *, timeout: timedelta) -> ytdlp_module.TrackMetadata:
        return ytdlp_module.TrackMetadata(artists=('Artist 1',), title='Song')

    monkeypatch.setattr(ytdlp_module, '_download_cover_as_jpg', _fake_cover)
    monkeypatch.setattr(ytdlp_module, '_download_track_metadata', _fake_metadata)

    result = await ytdlp_module.fetch_track_info(
        'https://example.com/watch?v=abc',
        with_cover=False,
        with_metadata=True,
    )

    assert result == ytdlp_module.UrlTrackInfo(
        cover=None,
        metadata=ytdlp_module.TrackMetadata(artists=('Artist 1',), title='Song'),
    )
    assert observed['cover_called'] is False


@pytest.mark.asyncio
async def test_fetch_track_info_with_no_flags_returns_empty_and_spawns_no_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _unexpected_create_subprocess_exec(*args: str, **kwargs: object) -> object:
        raise AssertionError('yt-dlp should not be spawned when no outputs are requested')

    monkeypatch.setattr(ytdlp_module.asyncio, 'create_subprocess_exec', _unexpected_create_subprocess_exec)

    result = await ytdlp_module.fetch_track_info(
        'https://example.com/watch?v=abc',
        with_cover=False,
        with_metadata=False,
    )

    assert result == ytdlp_module.UrlTrackInfo()


@pytest.mark.asyncio
async def test_fetch_track_info_rejects_non_string_url() -> None:
    with pytest.raises(ValueError, match='url must be a string'):
        await ytdlp_module.fetch_track_info(123)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_fetch_track_info_rejects_blank_url() -> None:
    with pytest.raises(ValueError, match='url must not be empty'):
        await ytdlp_module.fetch_track_info('   ')


@pytest.mark.asyncio
async def test_fetch_track_info_with_cover_propagates_cover_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _failing_cover(url: str, *, timeout: timedelta) -> bytes:
        raise RuntimeError('yt-dlp did not produce cover output')

    monkeypatch.setattr(ytdlp_module, '_download_cover_as_jpg', _failing_cover)

    with pytest.raises(RuntimeError, match='yt-dlp did not produce cover output'):
        await ytdlp_module.fetch_track_info(
            'https://example.com/watch?v=abc',
            with_cover=True,
            with_metadata=False,
        )


@pytest.mark.asyncio
async def test_fetch_track_info_with_metadata_propagates_metadata_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _failing_metadata(url: str, *, timeout: timedelta) -> ytdlp_module.TrackMetadata:
        raise YtDlpMetadataError('yt-dlp produced incomplete metadata output')

    monkeypatch.setattr(ytdlp_module, '_download_track_metadata', _failing_metadata)

    with pytest.raises(YtDlpMetadataError, match='yt-dlp produced incomplete metadata output'):
        await ytdlp_module.fetch_track_info(
            'https://example.com/watch?v=abc',
            with_cover=False,
            with_metadata=True,
        )


@pytest.mark.asyncio
async def test_download_audio_as_opus_builds_expected_command_and_returns_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}
    expected_bytes = b'OggS-opus-bytes'

    class _FakeProc:
        def __init__(self) -> None:
            self.returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            args = observed['args']
            output_template = Path(str(args[args.index('-o') + 1]))
            output_path = output_template.with_suffix('.opus')
            output_path.write_bytes(expected_bytes)
            return b'', b''

        def kill(self) -> None:
            return None

        async def wait(self) -> int:
            return self.returncode

    async def _fake_create_subprocess_exec(*args: str, **kwargs: object) -> _FakeProc:
        observed['args'] = args
        observed['kwargs'] = kwargs
        return _FakeProc()

    monkeypatch.setattr(ytdlp_module.asyncio, 'create_subprocess_exec', _fake_create_subprocess_exec)

    result = await ytdlp_module.download_audio_as_opus(
        '  https://example.com/watch?v=abc  ',
        timeout=timedelta(seconds=7),
    )

    assert result.audio == expected_bytes
    assert result.cover is None
    args = observed['args']
    assert args[0] == 'yt-dlp'
    _assert_yt_dlp_common_flags(args)
    assert '-f' in args
    assert args[args.index('-f') + 1] == 'bestaudio'
    assert '--extract-audio' in args
    assert '--audio-format' in args
    assert 'opus' in args
    assert '--no-playlist' in args
    assert '--write-thumbnail' not in args
    assert '--convert-thumbnails' not in args
    assert '--write-info-json' not in args
    assert '-o' in args
    assert args[-1] == 'https://example.com/watch?v=abc'


@pytest.mark.asyncio
async def test_download_audio_as_opus_with_cover_builds_expected_command_and_returns_dataclass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}
    expected_audio = b'OggS-opus-bytes'
    expected_cover = b'jpg-cover-bytes'

    class _FakeProc:
        def __init__(self) -> None:
            self.returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            args = observed['args']
            output_template = Path(str(args[args.index('-o') + 1]))
            output_template.with_suffix('.opus').write_bytes(expected_audio)
            output_template.with_suffix('.jpg').write_bytes(expected_cover)
            return b'', b''

        def kill(self) -> None:
            return None

        async def wait(self) -> int:
            return self.returncode

    async def _fake_create_subprocess_exec(*args: str, **kwargs: object) -> _FakeProc:
        observed['args'] = args
        observed['kwargs'] = kwargs
        return _FakeProc()

    monkeypatch.setattr(ytdlp_module.asyncio, 'create_subprocess_exec', _fake_create_subprocess_exec)

    result = await ytdlp_module.download_audio_as_opus(
        'https://example.com/watch?v=abc',
        with_cover=True,
        timeout=timedelta(seconds=7),
    )

    assert result.audio == expected_audio
    assert result.cover == expected_cover
    args = observed['args']
    _assert_yt_dlp_common_flags(args)
    assert '--write-thumbnail' in args
    assert '--convert-thumbnails' in args
    assert '--write-info-json' not in args
    assert 'jpg' in args


@pytest.mark.asyncio
async def test_download_audio_as_opus_with_metadata_writes_info_json_and_returns_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}
    expected_audio = b'OggS-opus-bytes'

    class _FakeProc:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            args = observed['args']
            output_template = Path(str(args[args.index('-o') + 1]))
            output_template.with_suffix('.opus').write_bytes(expected_audio)
            output_template.with_suffix('.info.json').write_text('{"artists": ["A1", "A2"], "track": "Song"}')
            return b'', b''

        def kill(self) -> None:
            return None

        async def wait(self) -> int:
            return self.returncode

    async def _fake_create_subprocess_exec(*args: str, **kwargs: object) -> _FakeProc:
        observed['args'] = args
        return _FakeProc()

    monkeypatch.setattr(ytdlp_module.asyncio, 'create_subprocess_exec', _fake_create_subprocess_exec)

    result = await ytdlp_module.download_audio_as_opus('https://example.com/watch?v=abc', with_metadata=True)

    assert result.audio == expected_audio
    assert result.cover is None
    assert result.metadata == ytdlp_module.TrackMetadata(artists=('A1', 'A2'), title='Song')
    args = observed['args']
    _assert_yt_dlp_common_flags(args)
    assert '--write-info-json' in args


@pytest.mark.asyncio
async def test_download_audio_as_opus_with_cover_and_metadata_returns_all_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    class _FakeProc:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            args = observed['args']
            output_template = Path(str(args[args.index('-o') + 1]))
            output_template.with_suffix('.opus').write_bytes(b'OggS-opus-bytes')
            output_template.with_suffix('.jpg').write_bytes(b'jpg-cover-bytes')
            output_template.with_suffix('.info.json').write_text('{"artist": "A1, A2", "title": "Track"}')
            return b'', b''

        def kill(self) -> None:
            return None

        async def wait(self) -> int:
            return self.returncode

    async def _fake_create_subprocess_exec(*args: str, **kwargs: object) -> _FakeProc:
        observed['args'] = args
        return _FakeProc()

    monkeypatch.setattr(ytdlp_module.asyncio, 'create_subprocess_exec', _fake_create_subprocess_exec)

    result = await ytdlp_module.download_audio_as_opus(
        'https://example.com/watch?v=abc',
        with_cover=True,
        with_metadata=True,
    )

    assert result.audio == b'OggS-opus-bytes'
    assert result.cover == b'jpg-cover-bytes'
    assert result.metadata == ytdlp_module.TrackMetadata(artists=('A1', 'A2'), title='Track')
    args = observed['args']
    _assert_yt_dlp_common_flags(args)
    assert '--write-thumbnail' in args
    assert '--write-info-json' in args


@pytest.mark.asyncio
async def test_download_audio_as_opus_with_metadata_raises_when_metadata_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    class _FakeProc:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            args = observed['args']
            output_template = Path(str(args[args.index('-o') + 1]))
            output_template.with_suffix('.opus').write_bytes(b'OggS-opus-bytes')
            return b'', b''

        def kill(self) -> None:
            return None

        async def wait(self) -> int:
            return self.returncode

    async def _fake_create_subprocess_exec(*args: str, **kwargs: object) -> _FakeProc:
        observed['args'] = args
        return _FakeProc()

    monkeypatch.setattr(ytdlp_module.asyncio, 'create_subprocess_exec', _fake_create_subprocess_exec)

    with pytest.raises(YtDlpMetadataError, match='yt-dlp did not produce metadata output'):
        await ytdlp_module.download_audio_as_opus('https://example.com/watch?v=abc', with_metadata=True)


@pytest.mark.asyncio
async def test_download_audio_as_opus_with_metadata_raises_when_multiple_metadata_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    class _FakeProc:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            args = observed['args']
            output_template = Path(str(args[args.index('-o') + 1]))
            output_template.with_suffix('.opus').write_bytes(b'OggS-opus-bytes')
            output_template.with_name('a.info.json').write_text('{}')
            output_template.with_name('b.info.json').write_text('{}')
            return b'', b''

        def kill(self) -> None:
            return None

        async def wait(self) -> int:
            return self.returncode

    async def _fake_create_subprocess_exec(*args: str, **kwargs: object) -> _FakeProc:
        observed['args'] = args
        return _FakeProc()

    monkeypatch.setattr(ytdlp_module.asyncio, 'create_subprocess_exec', _fake_create_subprocess_exec)

    with pytest.raises(YtDlpMetadataError, match='yt-dlp produced multiple metadata outputs'):
        await ytdlp_module.download_audio_as_opus('https://example.com/watch?v=abc', with_metadata=True)


@pytest.mark.asyncio
async def test_download_audio_as_opus_with_metadata_raises_when_metadata_invalid_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    class _FakeProc:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            args = observed['args']
            output_template = Path(str(args[args.index('-o') + 1]))
            output_template.with_suffix('.opus').write_bytes(b'OggS-opus-bytes')
            output_template.with_suffix('.info.json').write_text('{invalid json')
            return b'', b''

        def kill(self) -> None:
            return None

        async def wait(self) -> int:
            return self.returncode

    async def _fake_create_subprocess_exec(*args: str, **kwargs: object) -> _FakeProc:
        observed['args'] = args
        return _FakeProc()

    monkeypatch.setattr(ytdlp_module.asyncio, 'create_subprocess_exec', _fake_create_subprocess_exec)

    with pytest.raises(YtDlpMetadataError, match='yt-dlp produced invalid metadata output'):
        await ytdlp_module.download_audio_as_opus('https://example.com/watch?v=abc', with_metadata=True)


@pytest.mark.asyncio
async def test_download_audio_as_opus_with_metadata_raises_when_metadata_is_not_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    class _FakeProc:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            args = observed['args']
            output_template = Path(str(args[args.index('-o') + 1]))
            output_template.with_suffix('.opus').write_bytes(b'OggS-opus-bytes')
            output_template.with_suffix('.info.json').write_text('["not","an","object"]')
            return b'', b''

        def kill(self) -> None:
            return None

        async def wait(self) -> int:
            return self.returncode

    async def _fake_create_subprocess_exec(*args: str, **kwargs: object) -> _FakeProc:
        observed['args'] = args
        return _FakeProc()

    monkeypatch.setattr(ytdlp_module.asyncio, 'create_subprocess_exec', _fake_create_subprocess_exec)
    with pytest.raises(YtDlpMetadataError, match='yt-dlp produced invalid metadata output'):
        await ytdlp_module.download_audio_as_opus('https://example.com/watch?v=abc', with_metadata=True)


@pytest.mark.asyncio
async def test_download_audio_as_opus_with_metadata_uses_creator_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    class _FakeProc:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            args = observed['args']
            output_template = Path(str(args[args.index('-o') + 1]))
            output_template.with_suffix('.opus').write_bytes(b'OggS-opus-bytes')
            output_template.with_suffix('.info.json').write_text('{"creator":"C1, C2","title":"Track"}')
            return b'', b''

        def kill(self) -> None:
            return None

        async def wait(self) -> int:
            return self.returncode

    async def _fake_create_subprocess_exec(*args: str, **kwargs: object) -> _FakeProc:
        observed['args'] = args
        return _FakeProc()

    monkeypatch.setattr(ytdlp_module.asyncio, 'create_subprocess_exec', _fake_create_subprocess_exec)
    result = await ytdlp_module.download_audio_as_opus('https://example.com/watch?v=abc', with_metadata=True)
    assert result.metadata == ytdlp_module.TrackMetadata(artists=('C1', 'C2'), title='Track')


@pytest.mark.asyncio
async def test_download_audio_as_opus_with_metadata_title_falls_back_to_title_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    class _FakeProc:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            args = observed['args']
            output_template = Path(str(args[args.index('-o') + 1]))
            output_template.with_suffix('.opus').write_bytes(b'OggS-opus-bytes')
            output_template.with_suffix('.info.json').write_text('{"artists":["A1"],"title":"Fallback"}')
            return b'', b''

        def kill(self) -> None:
            return None

        async def wait(self) -> int:
            return self.returncode

    async def _fake_create_subprocess_exec(*args: str, **kwargs: object) -> _FakeProc:
        observed['args'] = args
        return _FakeProc()

    monkeypatch.setattr(ytdlp_module.asyncio, 'create_subprocess_exec', _fake_create_subprocess_exec)
    result = await ytdlp_module.download_audio_as_opus('https://example.com/watch?v=abc', with_metadata=True)
    assert result.metadata == ytdlp_module.TrackMetadata(artists=('A1',), title='Fallback')


@pytest.mark.asyncio
async def test_download_audio_as_opus_with_metadata_invalid_artists_list_falls_back_to_artist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    class _FakeProc:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            args = observed['args']
            output_template = Path(str(args[args.index('-o') + 1]))
            output_template.with_suffix('.opus').write_bytes(b'OggS-opus-bytes')
            output_template.with_suffix('.info.json').write_text(
                '{"artists":[123," "],"artist":"A1, A2","track":"Song"}'
            )
            return b'', b''

        def kill(self) -> None:
            return None

        async def wait(self) -> int:
            return self.returncode

    async def _fake_create_subprocess_exec(*args: str, **kwargs: object) -> _FakeProc:
        observed['args'] = args
        return _FakeProc()

    monkeypatch.setattr(ytdlp_module.asyncio, 'create_subprocess_exec', _fake_create_subprocess_exec)
    result = await ytdlp_module.download_audio_as_opus('https://example.com/watch?v=abc', with_metadata=True)
    assert result.metadata == ytdlp_module.TrackMetadata(artists=('A1', 'A2'), title='Song')


@pytest.mark.asyncio
async def test_download_audio_as_opus_with_metadata_raises_when_title_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    class _FakeProc:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            args = observed['args']
            output_template = Path(str(args[args.index('-o') + 1]))
            output_template.with_suffix('.opus').write_bytes(b'OggS-opus-bytes')
            output_template.with_suffix('.info.json').write_text('{"artists":["A1"]}')
            return b'', b''

        def kill(self) -> None:
            return None

        async def wait(self) -> int:
            return self.returncode

    async def _fake_create_subprocess_exec(*args: str, **kwargs: object) -> _FakeProc:
        observed['args'] = args
        return _FakeProc()

    monkeypatch.setattr(ytdlp_module.asyncio, 'create_subprocess_exec', _fake_create_subprocess_exec)
    with pytest.raises(YtDlpMetadataError, match='yt-dlp produced incomplete metadata output'):
        await ytdlp_module.download_audio_as_opus('https://example.com/watch?v=abc', with_metadata=True)


@pytest.mark.asyncio
async def test_download_audio_as_opus_with_metadata_raises_when_artists_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    class _FakeProc:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            args = observed['args']
            output_template = Path(str(args[args.index('-o') + 1]))
            output_template.with_suffix('.opus').write_bytes(b'OggS-opus-bytes')
            output_template.with_suffix('.info.json').write_text('{"track":"Song"}')
            return b'', b''

        def kill(self) -> None:
            return None

        async def wait(self) -> int:
            return self.returncode

    async def _fake_create_subprocess_exec(*args: str, **kwargs: object) -> _FakeProc:
        observed['args'] = args
        return _FakeProc()

    monkeypatch.setattr(ytdlp_module.asyncio, 'create_subprocess_exec', _fake_create_subprocess_exec)
    with pytest.raises(YtDlpMetadataError, match='yt-dlp produced incomplete metadata output'):
        await ytdlp_module.download_audio_as_opus('https://example.com/watch?v=abc', with_metadata=True)


@pytest.mark.asyncio
async def test_download_audio_as_opus_with_cover_raises_when_cover_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    class _FakeProc:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            args = observed['args']
            output_template = Path(str(args[args.index('-o') + 1]))
            output_template.with_suffix('.opus').write_bytes(b'OggS-opus-bytes')
            return b'', b''

        def kill(self) -> None:
            return None

        async def wait(self) -> int:
            return self.returncode

    async def _fake_create_subprocess_exec(*args: str, **kwargs: object) -> _FakeProc:
        observed['args'] = args
        return _FakeProc()

    monkeypatch.setattr(ytdlp_module.asyncio, 'create_subprocess_exec', _fake_create_subprocess_exec)

    with pytest.raises(RuntimeError, match='yt-dlp did not produce cover output'):
        await ytdlp_module.download_audio_as_opus('https://example.com/watch?v=abc', with_cover=True)


@pytest.mark.asyncio
async def test_download_audio_as_opus_rejects_non_string_url() -> None:
    with pytest.raises(ValueError, match='url must be a string'):
        await ytdlp_module.download_audio_as_opus(123)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_download_audio_as_opus_rejects_blank_url() -> None:
    with pytest.raises(ValueError, match='url must not be empty'):
        await ytdlp_module.download_audio_as_opus('   ')


@pytest.mark.asyncio
async def test_download_audio_as_opus_raises_runtime_error_on_non_zero_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeProc:
        returncode = 1

        async def communicate(self) -> tuple[bytes, bytes]:
            return b'', b'failure details'

        def kill(self) -> None:
            return None

        async def wait(self) -> int:
            return self.returncode

    async def _fake_create_subprocess_exec(*args: str, **kwargs: object) -> _FakeProc:
        return _FakeProc()

    monkeypatch.setattr(ytdlp_module.asyncio, 'create_subprocess_exec', _fake_create_subprocess_exec)

    with pytest.raises(RuntimeError, match=r'yt-dlp audio_download failed with exit code 1'):
        await ytdlp_module.download_audio_as_opus('https://example.com/watch?v=abc')


def test_sanitize_yt_dlp_args_redacts_sensitive_flags_and_sanitizes_urls() -> None:
    sanitized = ytdlp_module._sanitize_yt_dlp_args(
        (
            'yt-dlp',
            '--cookies',
            '/tmp/cookies.txt',
            '--proxy=http://user:pass@example.com:8080',
            '--username=alice',
            '--password',
            'hunter2',
            '--add-headers',
            'Authorization: Bearer secret',
            '--netrc-location=/home/user/.netrc',
            'https://example.com/watch?v=abc#frag',
            'plain-arg',
        )
    )

    assert sanitized == [
        'yt-dlp',
        '--cookies',
        '<redacted>',
        '--proxy=<redacted>',
        '--username=<redacted>',
        '--password',
        '<redacted>',
        '--add-headers',
        '<redacted>',
        '--netrc-location=<redacted>',
        'https://example.com/watch',
        'plain-arg',
    ]


def test_format_yt_dlp_failure_truncates_diagnostic_text() -> None:
    long_stderr = (
        'stderr-line-' * 320 + 'Sign in to confirm your age. Use --cookies-from-browser or --cookies.'
    ).encode()
    long_stdout = ('stdout-line-' * 600).encode()
    message = ytdlp_module._format_yt_dlp_failure(
        operation='audio_download',
        returncode=1,
        args=(
            'yt-dlp',
            '--username=alice',
            'https://example.com/watch?v=abc#frag',
        ),
        stderr=long_stderr,
        stdout=long_stdout,
    )

    assert 'yt-dlp audio_download failed with exit code 1' in message
    assert '--username=<redacted>' in message
    assert 'https://example.com/watch' in message
    assert 'Sign in to confirm your age. Use --cookies-from-browser or --cookies.' in message
    assert '... <truncated ' in message
    assert len(message) < 5000


def test_truncate_diagnostic_text_leaves_short_text_unchanged() -> None:
    assert ytdlp_module._truncate_diagnostic_text('short text', max_length=80) == 'short text'


def test_truncate_diagnostic_text_preserves_prefix_and_suffix() -> None:
    text = 'prefix-' + ('middle-' * 200) + 'final auth phrase'
    truncated = ytdlp_module._truncate_diagnostic_text(text, max_length=120)

    assert truncated.startswith('prefix-')
    assert truncated.endswith('final auth phrase')
    assert '... <truncated ' in truncated
    assert len(truncated) <= 120


@pytest.mark.parametrize(
    ('stderr', 'is_authentication_failure'),
    [
        (b"Sign in to confirm you're not a bot", True),
        (b'ERROR: [youtube] Please sign in to YouTube', True),
        (b'ERROR: [youtube] Login required for this request', True),
        (b'authentication cookies are no longer valid', True),
        (b'login required to access the internal dashboard', False),
        (b'invalid cookies received from an unrelated API', False),
        (b'expired cookies were removed from a local cache', False),
        (b'Sign in to confirm your age', False),
        (b'Video unavailable', False),
        (b'This video is private', False),
        (b'Requested format is not available', False),
        (b'network is unreachable', False),
    ],
)
def test_youtube_authentication_classifier_is_narrow(stderr: bytes, is_authentication_failure: bool) -> None:
    assert ytdlp_module._is_youtube_authentication_failure(stderr) is is_authentication_failure


@pytest.mark.asyncio
async def test_run_yt_dlp_command_raises_diagnostic_failure_without_logging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, tuple[str, ...]] = {}

    class _FakeProc:
        returncode = 2

        async def communicate(self) -> tuple[bytes, bytes]:
            return b'stdout payload', b'stderr payload'

        def kill(self) -> None:
            return None

        async def wait(self) -> int:
            return self.returncode

    async def _fake_create_subprocess_exec(*args: str, **kwargs: object) -> _FakeProc:
        observed['args'] = args
        return _FakeProc()

    with patch.object(ytdlp_module.logger, 'error') as error_mock:
        monkeypatch.setattr(ytdlp_module.asyncio, 'create_subprocess_exec', _fake_create_subprocess_exec)

        with pytest.raises(RuntimeError, match=r'yt-dlp metadata_download failed with exit code 2') as exc_info:
            await ytdlp_module._run_yt_dlp_command(
                operation='metadata_download',
                args=(
                    'yt-dlp',
                    '--username=alice',
                    'https://example.com/watch?v=abc#frag',
                ),
                timeout=timedelta(seconds=5),
            )

    error_mock.assert_not_called()
    message = str(exc_info.value)
    assert '--username=<redacted>' in message
    assert 'https://example.com/watch' in message
    assert 'stderr payload' in message
    assert 'stdout payload' in message


@pytest.mark.asyncio
async def test_shared_runner_raises_typed_authentication_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeProc:
        returncode = 1

        async def communicate(self) -> tuple[bytes, bytes]:
            return b'', b"Sign in to confirm you're not a bot"

        def kill(self) -> None:
            return None

        async def wait(self) -> int:
            return self.returncode

    async def _fake_create_subprocess_exec(*args: str, **kwargs: object) -> _FakeProc:
        return _FakeProc()

    monkeypatch.setattr(ytdlp_module.asyncio, 'create_subprocess_exec', _fake_create_subprocess_exec)

    with patch.object(ytdlp_module.logger, 'error') as error_mock:
        with pytest.raises(YtDlpAuthenticationError):
            await ytdlp_module._run_yt_dlp_command(
                operation='duration_probe',
                args=('yt-dlp', '--cookies', str(_COOKIE_FILE)),
                timeout=timedelta(seconds=5),
            )

    error_mock.assert_not_called()


@pytest.mark.asyncio
async def test_download_audio_as_opus_raises_diagnostic_timeout_without_logging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = {'killed': False, 'wait_calls': 0}
    wait_release = asyncio.Event()

    class _FakeProc:
        returncode = None

        async def communicate(self) -> tuple[bytes, bytes]:
            await asyncio.Event().wait()
            return b'', b''

        def kill(self) -> None:
            observed['killed'] = True
            wait_release.set()

        async def wait(self) -> int:
            observed['wait_calls'] += 1
            await wait_release.wait()
            self.returncode = -9
            return self.returncode

    async def _fake_create_subprocess_exec(*args: str, **kwargs: object) -> _FakeProc:
        return _FakeProc()

    real_wait_for = ytdlp_module.asyncio.wait_for

    async def _fake_wait_for(awaitable: object, timeout: float) -> object:
        del timeout
        return await real_wait_for(awaitable, 0.01)

    monkeypatch.setattr(ytdlp_module.asyncio, 'create_subprocess_exec', _fake_create_subprocess_exec)
    monkeypatch.setattr(ytdlp_module.asyncio, 'wait_for', _fake_wait_for)

    with patch.object(ytdlp_module.logger, 'error') as error_mock:
        with pytest.raises(asyncio.TimeoutError) as exc_info:
            await ytdlp_module.download_audio_as_opus(
                'https://example.com/watch?v=abc',
                timeout=timedelta(seconds=3),
            )

    assert observed['killed'] is True
    assert observed['wait_calls'] == 1
    error_mock.assert_not_called()
    assert 'yt-dlp audio_download timed out after 3.0s' in str(exc_info.value)
    assert 'command=[' in str(exc_info.value)
    assert 'https://example.com/watch' in str(exc_info.value)


@pytest.mark.asyncio
async def test_cancelling_shared_runner_reaps_child_before_removing_operation_cookie(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical_cookie_file = tmp_path / 'youtube-cookies.txt'
    canonical_cookie_file.write_bytes(_COOKIE_DATA)
    communication_started = asyncio.Event()
    terminated = asyncio.Event()
    allow_reap = asyncio.Event()
    observed: dict[str, object] = {'terminated': False, 'killed': False, 'waited': False}

    class _FakeProc:
        returncode: int | None = None

        async def communicate(self) -> tuple[bytes, bytes]:
            communication_started.set()
            await asyncio.Event().wait()
            return b'', b''

        def terminate(self) -> None:
            observed['terminated'] = True
            terminated.set()

        def kill(self) -> None:
            observed['killed'] = True
            allow_reap.set()

        async def wait(self) -> int:
            await allow_reap.wait()
            operation_cookie_file = observed['operation_cookie_file']
            assert isinstance(operation_cookie_file, Path)
            assert operation_cookie_file.is_file()
            self.returncode = 0
            observed['waited'] = True
            return self.returncode

    async def _fake_create_subprocess_exec(*args: str, **kwargs: object) -> _FakeProc:
        observed['operation_cookie_file'] = Path(args[args.index('--cookies') + 1])
        return _FakeProc()

    monkeypatch.setattr(ytdlp_module.asyncio, 'create_subprocess_exec', _fake_create_subprocess_exec)
    operation = asyncio.create_task(
        ytdlp_module.get_media_duration(
            'https://example.com/watch?v=abc',
            cookie_file=canonical_cookie_file,
        ),
    )
    await communication_started.wait()
    operation.cancel()
    await terminated.wait()

    operation_cookie_file = observed['operation_cookie_file']
    assert isinstance(operation_cookie_file, Path)
    assert operation_cookie_file.is_file()
    allow_reap.set()
    with patch.object(ytdlp_module.logger, 'error') as error_mock:
        with pytest.raises(asyncio.CancelledError):
            await operation

    assert observed['terminated'] is True
    assert observed['killed'] is False
    assert observed['waited'] is True
    assert not operation_cookie_file.exists()
    error_mock.assert_not_called()


@pytest.mark.asyncio
async def test_cancelling_shared_runner_kills_child_after_grace_period(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical_cookie_file = tmp_path / 'youtube-cookies.txt'
    canonical_cookie_file.write_bytes(_COOKIE_DATA)
    communication_started = asyncio.Event()
    killed = asyncio.Event()
    observed = {'terminated': False, 'killed': False, 'waited': False}

    class _FakeProc:
        returncode: int | None = None

        async def communicate(self) -> tuple[bytes, bytes]:
            communication_started.set()
            await asyncio.Event().wait()
            return b'', b''

        def terminate(self) -> None:
            observed['terminated'] = True

        def kill(self) -> None:
            observed['killed'] = True
            self.returncode = -9
            killed.set()

        async def wait(self) -> int:
            await killed.wait()
            observed['waited'] = True
            return self.returncode or -9

    async def _fake_create_subprocess_exec(*args: str, **kwargs: object) -> _FakeProc:
        return _FakeProc()

    monkeypatch.setattr(ytdlp_module, '_PROCESS_TERMINATION_GRACE_SECONDS', 0.01)
    monkeypatch.setattr(ytdlp_module.asyncio, 'create_subprocess_exec', _fake_create_subprocess_exec)
    operation = asyncio.create_task(
        ytdlp_module.get_media_duration(
            'https://example.com/watch?v=abc',
            cookie_file=canonical_cookie_file,
        ),
    )
    await communication_started.wait()
    operation.cancel()

    with patch.object(ytdlp_module.logger, 'error') as error_mock:
        with pytest.raises(asyncio.CancelledError):
            await operation

    assert observed == {'terminated': True, 'killed': True, 'waited': True}
    error_mock.assert_not_called()


@pytest.mark.asyncio
async def test_cancelling_subprocess_creation_reaps_created_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    creation_started = asyncio.Event()
    allow_creation = asyncio.Event()
    observed = {'terminated': False, 'waited': False}

    class _FakeProc:
        returncode: int | None = None

        def terminate(self) -> None:
            observed['terminated'] = True
            self.returncode = 0

        def kill(self) -> None:
            self.returncode = -9

        async def wait(self) -> int:
            observed['waited'] = True
            return self.returncode or 0

    async def _fake_create_subprocess_exec(*args: str, **kwargs: object) -> _FakeProc:
        creation_started.set()
        await allow_creation.wait()
        return _FakeProc()

    monkeypatch.setattr(ytdlp_module.asyncio, 'create_subprocess_exec', _fake_create_subprocess_exec)
    creation = asyncio.create_task(ytdlp_module._create_owned_subprocess_exec('yt-dlp'))
    await creation_started.wait()
    creation.cancel()
    allow_creation.set()

    with patch.object(ytdlp_module.logger, 'error') as error_mock:
        with pytest.raises(asyncio.CancelledError):
            await creation

    assert observed == {'terminated': True, 'waited': True}
    error_mock.assert_not_called()


@pytest.mark.asyncio
async def test_cancelling_subprocess_creation_ignores_spawn_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    creation_started = asyncio.Event()
    allow_failure = asyncio.Event()

    async def _fake_create_subprocess_exec(*args: str, **kwargs: object) -> None:
        creation_started.set()
        await allow_failure.wait()
        raise OSError('spawn failed')

    async def _unexpected_stop_process(process: asyncio.subprocess.Process) -> None:
        raise AssertionError('no child exists to stop')

    monkeypatch.setattr(ytdlp_module.asyncio, 'create_subprocess_exec', _fake_create_subprocess_exec)
    monkeypatch.setattr(ytdlp_module, '_stop_process', _unexpected_stop_process)
    creation = asyncio.create_task(ytdlp_module._create_owned_subprocess_exec('yt-dlp'))
    await creation_started.wait()
    creation.cancel()
    allow_failure.set()

    with patch.object(ytdlp_module.logger, 'warning') as warning_mock:
        with pytest.raises(asyncio.CancelledError):
            await creation

    warning_mock.assert_not_called()


@pytest.mark.asyncio
async def test_download_audio_as_opus_raises_when_no_opus_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeProc:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return b'', b''

        def kill(self) -> None:
            return None

        async def wait(self) -> int:
            return self.returncode

    async def _fake_create_subprocess_exec(*args: str, **kwargs: object) -> _FakeProc:
        return _FakeProc()

    monkeypatch.setattr(ytdlp_module.asyncio, 'create_subprocess_exec', _fake_create_subprocess_exec)

    with pytest.raises(RuntimeError, match='yt-dlp did not produce opus output'):
        await ytdlp_module.download_audio_as_opus('https://example.com/watch?v=abc')


@pytest.mark.asyncio
async def test_download_audio_as_opus_raises_when_multiple_opus_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    class _FakeProc:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            args = observed['args']
            output_template = Path(str(args[args.index('-o') + 1]))
            output_template.with_name('audio1.opus').write_bytes(b'1')
            output_template.with_name('audio2.opus').write_bytes(b'2')
            return b'', b''

        def kill(self) -> None:
            return None

        async def wait(self) -> int:
            return self.returncode

    async def _fake_create_subprocess_exec(*args: str, **kwargs: object) -> _FakeProc:
        observed['args'] = args
        return _FakeProc()

    monkeypatch.setattr(ytdlp_module.asyncio, 'create_subprocess_exec', _fake_create_subprocess_exec)

    with pytest.raises(RuntimeError, match='yt-dlp produced multiple opus outputs'):
        await ytdlp_module.download_audio_as_opus('https://example.com/watch?v=abc')


@pytest.mark.asyncio
async def test_download_audio_as_opus_raises_when_output_is_not_ogg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    class _FakeProc:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            args = observed['args']
            output_template = Path(str(args[args.index('-o') + 1]))
            output_path = output_template.with_suffix('.opus')
            output_path.write_bytes(b'not-ogg-data')
            return b'', b''

        def kill(self) -> None:
            return None

        async def wait(self) -> int:
            return self.returncode

    async def _fake_create_subprocess_exec(*args: str, **kwargs: object) -> _FakeProc:
        observed['args'] = args
        return _FakeProc()

    monkeypatch.setattr(ytdlp_module.asyncio, 'create_subprocess_exec', _fake_create_subprocess_exec)

    with pytest.raises(RuntimeError, match='yt-dlp output is not a valid Ogg/Opus container'):
        await ytdlp_module.download_audio_as_opus('https://example.com/watch?v=abc')


@pytest.mark.asyncio
async def test_get_media_duration_returns_timedelta_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    class _FakeProc:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return b'123.45\n', b''

        def kill(self) -> None:
            return None

        async def wait(self) -> int:
            return self.returncode

    async def _fake_create_subprocess_exec(*args: str, **kwargs: object) -> _FakeProc:
        observed['args'] = args
        return _FakeProc()

    monkeypatch.setattr(ytdlp_module.asyncio, 'create_subprocess_exec', _fake_create_subprocess_exec)

    duration = await ytdlp_module.get_media_duration('https://example.com/watch?v=abc')

    assert duration == timedelta(seconds=123.45)
    args = observed['args']
    assert args[0] == 'yt-dlp'
    _assert_yt_dlp_common_flags(args)
    assert '--print' in args
    assert '%(duration)s' in args
    assert '--simulate' in args
    assert '--skip-download' in args
    assert '--ignore-no-formats-error' in args
    assert '--no-playlist' in args


@pytest.mark.asyncio
async def test_get_media_duration_returns_none_on_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeProc:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return b'NA\n', b''

        def kill(self) -> None:
            return None

        async def wait(self) -> int:
            return self.returncode

    async def _fake_create_subprocess_exec(*args: str, **kwargs: object) -> _FakeProc:
        return _FakeProc()

    monkeypatch.setattr(ytdlp_module.asyncio, 'create_subprocess_exec', _fake_create_subprocess_exec)

    duration = await ytdlp_module.get_media_duration('https://example.com/watch?v=abc')

    assert duration is None


@pytest.mark.asyncio
async def test_get_media_duration_returns_value_when_no_formats_are_selectable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    class _FakeProc:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return b'42.0\n', b'Requested format is not available\n'

        def kill(self) -> None:
            return None

        async def wait(self) -> int:
            return self.returncode

    async def _fake_create_subprocess_exec(*args: str, **kwargs: object) -> _FakeProc:
        observed['args'] = args
        return _FakeProc()

    monkeypatch.setattr(ytdlp_module.asyncio, 'create_subprocess_exec', _fake_create_subprocess_exec)

    duration = await ytdlp_module.get_media_duration('https://example.com/watch?v=no-formats')

    assert duration == timedelta(seconds=42)
    args = observed['args']
    assert '--simulate' in args
    assert '--ignore-no-formats-error' in args


@pytest.mark.asyncio
async def test_get_media_duration_raises_on_non_zero_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeProc:
        returncode = 1

        async def communicate(self) -> tuple[bytes, bytes]:
            return b'', b'probe error'

        def kill(self) -> None:
            return None

        async def wait(self) -> int:
            return self.returncode

    async def _fake_create_subprocess_exec(*args: str, **kwargs: object) -> _FakeProc:
        return _FakeProc()

    monkeypatch.setattr(ytdlp_module.asyncio, 'create_subprocess_exec', _fake_create_subprocess_exec)

    with pytest.raises(RuntimeError, match=r'yt-dlp duration_probe failed with exit code 1'):
        await ytdlp_module.get_media_duration('https://example.com/watch?v=abc')


@pytest.mark.asyncio
async def test_download_audio_as_opus_with_max_duration_uses_full_path_for_short_media(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = {'clipped_called': False}

    async def _fake_get_media_duration(url: str, *, timeout: timedelta) -> timedelta | None:
        return timedelta(seconds=29)

    async def _fake_internal(
        url: str,
        *,
        download_cover: bool,
        with_metadata: bool,
        timeout: timedelta,
    ) -> tuple[bytes, bytes | None, ytdlp_module.TrackMetadata | None]:
        return b'full', None, None

    async def _fake_clipped(
        url: str,
        *,
        max_duration: timedelta,
        timeout: timedelta,
    ) -> bytes:
        observed['clipped_called'] = True
        return b'clipped'

    monkeypatch.setattr(ytdlp_module, '_get_media_duration', _fake_get_media_duration)
    monkeypatch.setattr(ytdlp_module, '_download_audio_as_opus_internal', _fake_internal)
    monkeypatch.setattr(ytdlp_module, '_download_audio_as_opus_clipped', _fake_clipped)

    result = await ytdlp_module.download_audio_as_opus(
        'https://example.com/watch?v=abc',
        max_duration=timedelta(seconds=30),
    )

    assert result.audio == b'full'
    assert result.cover is None
    assert observed['clipped_called'] is False


@pytest.mark.asyncio
async def test_download_audio_as_opus_with_max_duration_uses_clipped_path_for_long_media(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = {'full_called': False}

    async def _fake_get_media_duration(url: str, *, timeout: timedelta) -> timedelta | None:
        return timedelta(seconds=31)

    async def _fake_internal(
        url: str,
        *,
        download_cover: bool,
        with_metadata: bool,
        timeout: timedelta,
    ) -> tuple[bytes, bytes | None, ytdlp_module.TrackMetadata | None]:
        observed['full_called'] = True
        return b'full', None, None

    async def _fake_clipped(
        url: str,
        *,
        max_duration: timedelta,
        timeout: timedelta,
    ) -> bytes:
        return b'clipped'

    monkeypatch.setattr(ytdlp_module, '_get_media_duration', _fake_get_media_duration)
    monkeypatch.setattr(ytdlp_module, '_download_audio_as_opus_internal', _fake_internal)
    monkeypatch.setattr(ytdlp_module, '_download_audio_as_opus_clipped', _fake_clipped)

    result = await ytdlp_module.download_audio_as_opus(
        'https://example.com/watch?v=abc',
        max_duration=timedelta(seconds=30),
    )

    assert result.audio == b'clipped'
    assert result.cover is None
    assert observed['full_called'] is False


@pytest.mark.asyncio
async def test_download_audio_as_opus_with_max_duration_uses_clipped_path_for_unknown_duration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = {'full_called': False}

    async def _fake_get_media_duration(url: str, *, timeout: timedelta) -> timedelta | None:
        return None

    async def _fake_internal(
        url: str,
        *,
        download_cover: bool,
        with_metadata: bool,
        timeout: timedelta,
    ) -> tuple[bytes, bytes | None, ytdlp_module.TrackMetadata | None]:
        observed['full_called'] = True
        return b'full', None, None

    async def _fake_clipped(
        url: str,
        *,
        max_duration: timedelta,
        timeout: timedelta,
    ) -> bytes:
        return b'clipped'

    monkeypatch.setattr(ytdlp_module, '_get_media_duration', _fake_get_media_duration)
    monkeypatch.setattr(ytdlp_module, '_download_audio_as_opus_internal', _fake_internal)
    monkeypatch.setattr(ytdlp_module, '_download_audio_as_opus_clipped', _fake_clipped)

    result = await ytdlp_module.download_audio_as_opus(
        'https://example.com/watch?v=abc',
        max_duration=timedelta(seconds=30),
    )

    assert result.audio == b'clipped'
    assert result.cover is None
    assert observed['full_called'] is False


@pytest.mark.asyncio
async def test_download_audio_as_opus_with_metadata_uses_clipped_audio_and_fetches_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    async def _fake_get_media_duration(url: str, *, timeout: timedelta) -> timedelta | None:
        return timedelta(seconds=31)

    async def _fake_clipped(
        url: str,
        *,
        max_duration: timedelta,
        timeout: timedelta,
    ) -> bytes:
        observed['clipped_called'] = True
        return b'OggS-clipped-audio'

    async def _fake_metadata(
        url: str,
        *,
        timeout: timedelta,
    ) -> ytdlp_module.TrackMetadata:
        observed['metadata_called'] = True
        return ytdlp_module.TrackMetadata(artists=('Artist 1',), title='Song 1')

    monkeypatch.setattr(ytdlp_module, '_get_media_duration', _fake_get_media_duration)
    monkeypatch.setattr(ytdlp_module, '_download_audio_as_opus_clipped', _fake_clipped)
    monkeypatch.setattr(ytdlp_module, '_download_track_metadata', _fake_metadata)

    result = await ytdlp_module.download_audio_as_opus(
        'https://example.com/watch?v=abc',
        with_metadata=True,
        max_duration=timedelta(seconds=30),
    )

    assert result.audio == b'OggS-clipped-audio'
    assert result.cover is None
    assert result.metadata == ytdlp_module.TrackMetadata(artists=('Artist 1',), title='Song 1')
    assert observed['clipped_called'] is True
    assert observed['metadata_called'] is True


@pytest.mark.asyncio
async def test_download_audio_as_opus_with_cover_and_metadata_and_max_duration_uses_clipped_audio_and_fetches_cover(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_get_media_duration(url: str, *, timeout: timedelta) -> timedelta | None:
        return timedelta(seconds=31)

    async def _fake_clipped(
        url: str,
        *,
        max_duration: timedelta,
        timeout: timedelta,
    ) -> bytes:
        return b'OggS-clipped-audio'

    async def _fake_metadata(
        url: str,
        *,
        timeout: timedelta,
    ) -> ytdlp_module.TrackMetadata:
        return ytdlp_module.TrackMetadata(artists=('Artist 1',), title='Song 1')

    async def _fake_cover(url: str, *, timeout: timedelta) -> bytes:
        return b'cover-jpg'

    monkeypatch.setattr(ytdlp_module, '_get_media_duration', _fake_get_media_duration)
    monkeypatch.setattr(ytdlp_module, '_download_audio_as_opus_clipped', _fake_clipped)
    monkeypatch.setattr(ytdlp_module, '_download_track_metadata', _fake_metadata)
    monkeypatch.setattr(ytdlp_module, '_download_cover_as_jpg', _fake_cover)

    result = await ytdlp_module.download_audio_as_opus(
        'https://example.com/watch?v=abc',
        with_cover=True,
        with_metadata=True,
        max_duration=timedelta(seconds=30),
    )

    assert result.audio == b'OggS-clipped-audio'
    assert result.cover == b'cover-jpg'
    assert result.metadata == ytdlp_module.TrackMetadata(artists=('Artist 1',), title='Song 1')


@pytest.mark.asyncio
async def test_download_audio_as_opus_with_cover_and_max_duration_uses_full_path_for_short_media(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = {'clipped_called': False}

    async def _fake_get_media_duration(url: str, *, timeout: timedelta) -> timedelta | None:
        return timedelta(seconds=29)

    async def _fake_internal(
        url: str,
        *,
        download_cover: bool,
        with_metadata: bool,
        timeout: timedelta,
    ) -> tuple[bytes, bytes | None, ytdlp_module.TrackMetadata | None]:
        assert download_cover is True
        return b'full-audio', b'full-cover', None

    async def _fake_clipped(
        url: str,
        *,
        max_duration: timedelta,
        timeout: timedelta,
    ) -> bytes:
        observed['clipped_called'] = True
        return b'clipped-audio'

    monkeypatch.setattr(ytdlp_module, '_get_media_duration', _fake_get_media_duration)
    monkeypatch.setattr(ytdlp_module, '_download_audio_as_opus_internal', _fake_internal)
    monkeypatch.setattr(ytdlp_module, '_download_audio_as_opus_clipped', _fake_clipped)

    result = await ytdlp_module.download_audio_as_opus(
        'https://example.com/watch?v=abc',
        with_cover=True,
        max_duration=timedelta(seconds=30),
    )

    assert result.audio == b'full-audio'
    assert result.cover == b'full-cover'
    assert observed['clipped_called'] is False


@pytest.mark.asyncio
async def test_download_audio_as_opus_with_cover_and_max_duration_uses_clipped_audio_and_thumbnail_fetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {'full_called': False}

    async def _fake_get_media_duration(url: str, *, timeout: timedelta) -> timedelta | None:
        return timedelta(seconds=31)

    async def _fake_internal(
        url: str,
        *,
        download_cover: bool,
        with_metadata: bool,
        timeout: timedelta,
    ) -> tuple[bytes, bytes | None, ytdlp_module.TrackMetadata | None]:
        observed['full_called'] = True
        return b'full-audio', b'full-cover', None

    async def _fake_clipped(
        url: str,
        *,
        max_duration: timedelta,
        timeout: timedelta,
    ) -> bytes:
        return b'OggS-clipped-audio'

    class _FakeProc:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            args = observed['thumb_args']
            output_template = Path(str(args[args.index('-o') + 1]))
            output_template.with_suffix('.jpg').write_bytes(b'thumb-jpg-bytes')
            return b'', b''

        def kill(self) -> None:
            return None

        async def wait(self) -> int:
            return self.returncode

    async def _fake_create_subprocess_exec(*args: str, **kwargs: object) -> _FakeProc:
        observed['thumb_args'] = args
        return _FakeProc()

    monkeypatch.setattr(ytdlp_module, '_get_media_duration', _fake_get_media_duration)
    monkeypatch.setattr(ytdlp_module, '_download_audio_as_opus_internal', _fake_internal)
    monkeypatch.setattr(ytdlp_module, '_download_audio_as_opus_clipped', _fake_clipped)
    monkeypatch.setattr(ytdlp_module.asyncio, 'create_subprocess_exec', _fake_create_subprocess_exec)

    result = await ytdlp_module.download_audio_as_opus(
        'https://example.com/watch?v=abc',
        with_cover=True,
        max_duration=timedelta(seconds=30),
    )

    assert result.audio == b'OggS-clipped-audio'
    assert result.cover == b'thumb-jpg-bytes'
    assert observed['full_called'] is False
    thumb_args = observed['thumb_args']
    assert thumb_args[0] == 'yt-dlp'
    assert '--skip-download' in thumb_args
    assert '--write-thumbnail' in thumb_args
    assert '--convert-thumbnails' in thumb_args
    assert 'jpg' in thumb_args


@pytest.mark.asyncio
async def test_download_audio_functions_do_not_probe_duration_when_max_duration_is_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _unexpected_get_media_duration(url: str, *, timeout: timedelta) -> timedelta | None:
        raise AssertionError('duration probe should not be called when max_duration is None')

    async def _fake_internal(
        url: str,
        *,
        download_cover: bool,
        with_metadata: bool,
        timeout: timedelta,
    ) -> tuple[bytes, bytes | None, ytdlp_module.TrackMetadata | None]:
        if download_cover:
            return b'full-audio', b'full-cover', None
        return b'full-audio', None, None

    monkeypatch.setattr(ytdlp_module, '_get_media_duration', _unexpected_get_media_duration)
    monkeypatch.setattr(ytdlp_module, '_download_audio_as_opus_internal', _fake_internal)

    audio = await ytdlp_module.download_audio_as_opus('https://example.com/watch?v=abc')
    audio_with_cover = await ytdlp_module.download_audio_as_opus(
        'https://example.com/watch?v=abc',
        with_cover=True,
    )

    assert audio.audio == b'full-audio'
    assert audio.cover is None
    assert audio_with_cover.audio == b'full-audio'
    assert audio_with_cover.cover == b'full-cover'


@pytest.mark.asyncio
async def test_download_audio_functions_reject_non_positive_max_duration() -> None:
    with pytest.raises(ValueError, match='max_duration must be > 0'):
        await ytdlp_module.download_audio_as_opus(
            'https://example.com/watch?v=abc',
            max_duration=timedelta(0),
        )
    with pytest.raises(ValueError, match='max_duration must be > 0'):
        await ytdlp_module.download_audio_as_opus(
            'https://example.com/watch?v=abc',
            max_duration=timedelta(seconds=-1),
        )
    with pytest.raises(ValueError, match='max_duration must be > 0'):
        await ytdlp_module.download_audio_as_opus(
            'https://example.com/watch?v=abc',
            max_duration=timedelta(seconds=-1),
            with_cover=True,
        )


@pytest.mark.asyncio
async def test_download_audio_as_opus_clipped_baseline_builds_pipeline_and_returns_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {'calls': []}

    class _FakeReader:
        def __init__(self, payload: bytes) -> None:
            self._payload = payload

        async def read(self) -> bytes:
            return self._payload

    class _CodecProbeProc:
        def __init__(self) -> None:
            self.returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return b'opus\n', b''

        def kill(self) -> None:
            return None

        async def wait(self) -> int:
            return self.returncode

    class _YtDlpProc:
        def __init__(self) -> None:
            self.returncode = 0
            self.stdout = object()
            self.stderr = _FakeReader(b'')
            self.communicate_called = False

        async def communicate(self) -> tuple[bytes, bytes]:
            self.communicate_called = True
            raise AssertionError('ytdlp communicate must not be called in clipped mode')

        def kill(self) -> None:
            return None

        async def wait(self) -> int:
            return self.returncode

    class _FfmpegProc:
        def __init__(self) -> None:
            self.returncode = 0
            self.stdin = object()
            self.stdout = _FakeReader(b'OggS-clipped')
            self.stderr = _FakeReader(b'')
            self.communicate_called = False

        async def communicate(self) -> tuple[bytes, bytes]:
            self.communicate_called = True
            raise AssertionError('ffmpeg communicate must not be called in clipped mode')

        def kill(self) -> None:
            return None

        async def wait(self) -> int:
            return self.returncode

    async def _fake_pipe_stream(source: object, destination: object) -> None:
        return None

    async def _fake_create_subprocess_exec(*args: str, **kwargs: object) -> object:
        observed['calls'].append(args)
        if '--print' in args:
            return _CodecProbeProc()
        if args[0] == 'yt-dlp':
            observed['yt-dlp'] = args
            return _YtDlpProc()
        observed['ffmpeg'] = args
        return _FfmpegProc()

    monkeypatch.setattr(ytdlp_module, '_pipe_stream', _fake_pipe_stream)
    monkeypatch.setattr(ytdlp_module.asyncio, 'create_subprocess_exec', _fake_create_subprocess_exec)

    result = await ytdlp_module._download_audio_as_opus_clipped(
        'https://example.com/watch?v=abc',
        max_duration=timedelta(seconds=15),
        timeout=timedelta(seconds=10),
    )

    assert result == b'OggS-clipped'
    assert len(observed['calls']) == 3
    codec_probe_args = observed['calls'][0]
    ytdlp_args = observed['calls'][1]
    ffmpeg_args = observed['ffmpeg']
    assert codec_probe_args[0] == 'yt-dlp'
    _assert_yt_dlp_common_flags(codec_probe_args)
    _assert_cookie_file_argument(codec_probe_args, _COOKIE_FILE)
    _assert_ejs_arguments(codec_probe_args)
    _assert_no_failed_experiment_arguments(codec_probe_args)
    assert '--print' in codec_probe_args
    assert '%(acodec)s' in codec_probe_args
    assert ytdlp_args[0] == 'yt-dlp'
    _assert_yt_dlp_common_flags(ytdlp_args)
    _assert_cookie_file_argument(ytdlp_args, _COOKIE_FILE)
    _assert_ejs_arguments(ytdlp_args)
    _assert_no_failed_experiment_arguments(ytdlp_args)
    assert ffmpeg_args[ffmpeg_args.index('-c:a') + 1] == 'copy'
    assert '-o' in ytdlp_args
    assert ytdlp_args[ytdlp_args.index('-o') + 1] == '-'
    assert '-f' in ytdlp_args
    assert ytdlp_args[ytdlp_args.index('-f') + 1] == 'bestaudio'
    assert 'pipe:0' in ffmpeg_args
    assert '-t' in ffmpeg_args
    assert ffmpeg_args[ffmpeg_args.index('-t') + 1] == '15.0'
    assert '-c:a' in ffmpeg_args
    assert '-f' in ffmpeg_args
    assert ffmpeg_args[ffmpeg_args.index('-f') + 1] == 'opus'
    assert ffmpeg_args[-1] == 'pipe:1'


@pytest.mark.asyncio
async def test_download_audio_as_opus_clipped_tolerates_ytdlp_broken_pipe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {'calls': [], 'ytdlp_waits': 0, 'ffmpeg_waits': 0}

    class _FakeReader:
        def __init__(self, payload: bytes) -> None:
            self._payload = payload

        async def read(self) -> bytes:
            return self._payload

    class _YtDlpProc:
        def __init__(self) -> None:
            self.returncode = 1
            self.stdout = object()
            self.stderr = _FakeReader(b'ERROR: Broken pipe')

        def kill(self) -> None:
            return None

        async def wait(self) -> int:
            observed['ytdlp_waits'] += 1
            return self.returncode

    class _FfmpegProc:
        def __init__(self) -> None:
            self.returncode = 0
            self.stdin = object()
            self.stdout = _FakeReader(b'OggS-clipped')
            self.stderr = _FakeReader(b'')
            self.communicate_called = False

        async def communicate(self) -> tuple[bytes, bytes]:
            self.communicate_called = True
            raise AssertionError('ffmpeg communicate must not be called in clipped mode')

        def kill(self) -> None:
            return None

        async def wait(self) -> int:
            observed['ffmpeg_waits'] += 1
            return self.returncode

    async def _fake_pipe_stream(source: object, destination: object) -> None:
        return None

    async def _fake_get_selected_audio_codec(url: str, *, timeout: timedelta) -> str | None:
        return 'opus'

    async def _fake_create_subprocess_exec(*args: str, **kwargs: object) -> object:
        if args[0] == 'yt-dlp':
            observed['yt-dlp'] = args
            return _YtDlpProc()
        observed['ffmpeg'] = args
        return _FfmpegProc()

    monkeypatch.setattr(ytdlp_module, '_pipe_stream', _fake_pipe_stream)
    monkeypatch.setattr(ytdlp_module, '_get_selected_audio_codec', _fake_get_selected_audio_codec)
    monkeypatch.setattr(ytdlp_module.asyncio, 'create_subprocess_exec', _fake_create_subprocess_exec)

    result = await ytdlp_module._download_audio_as_opus_clipped(
        'https://example.com/watch?v=abc',
        max_duration=timedelta(seconds=15),
        timeout=timedelta(seconds=10),
    )
    assert result == b'OggS-clipped'
    assert observed['ytdlp_waits'] == 1
    assert observed['ffmpeg_waits'] == 1


@pytest.mark.parametrize(
    (
        'ytdlp_stderr',
        'ffmpeg_returncode',
        'ffmpeg_stdout',
        'ffmpeg_stderr',
        'exception_type',
    ),
    [
        (b'ytdlp boom', 0, b'OggS-clipped', b'', RuntimeError),
        (b"Sign in to confirm you're not a bot", 0, b'OggS-clipped', b'', YtDlpAuthenticationError),
        (
            b"ERROR: [youtube] Sign in to confirm you're not a bot",
            1,
            b'',
            b'pipe input failed',
            YtDlpAuthenticationError,
        ),
        (b'ytdlp boom', 1, b'', b'pipe input failed', RuntimeError),
    ],
)
@pytest.mark.asyncio
async def test_download_audio_as_opus_clipped_raises_failures_without_infra_error_logging(
    monkeypatch: pytest.MonkeyPatch,
    ytdlp_stderr: bytes,
    ffmpeg_returncode: int,
    ffmpeg_stdout: bytes,
    ffmpeg_stderr: bytes,
    exception_type: type[RuntimeError],
) -> None:
    observed: dict[str, object] = {'calls': [], 'ytdlp_waits': 0, 'ffmpeg_waits': 0}

    class _FakeReader:
        def __init__(self, payload: bytes) -> None:
            self._payload = payload

        async def read(self) -> bytes:
            return self._payload

    class _YtDlpProc:
        def __init__(self) -> None:
            self.returncode = 2
            self.stdout = object()
            self.stderr = _FakeReader(ytdlp_stderr)

        def kill(self) -> None:
            return None

        async def wait(self) -> int:
            observed['ytdlp_waits'] += 1
            return self.returncode

    class _FfmpegProc:
        def __init__(self) -> None:
            self.returncode = ffmpeg_returncode
            self.stdin = object()
            self.stdout = _FakeReader(ffmpeg_stdout)
            self.stderr = _FakeReader(ffmpeg_stderr)

        def kill(self) -> None:
            return None

        async def wait(self) -> int:
            observed['ffmpeg_waits'] += 1
            return self.returncode

    async def _fake_pipe_stream(source: object, destination: object) -> None:
        return None

    async def _fake_get_selected_audio_codec(url: str, *, timeout: timedelta) -> str | None:
        return 'opus'

    async def _fake_create_subprocess_exec(*args: str, **kwargs: object) -> object:
        if args[0] == 'yt-dlp':
            return _YtDlpProc()
        return _FfmpegProc()

    monkeypatch.setattr(ytdlp_module, '_pipe_stream', _fake_pipe_stream)
    monkeypatch.setattr(ytdlp_module, '_get_selected_audio_codec', _fake_get_selected_audio_codec)
    monkeypatch.setattr(ytdlp_module.asyncio, 'create_subprocess_exec', _fake_create_subprocess_exec)

    with patch.object(ytdlp_module.logger, 'error') as error_mock:
        with pytest.raises(exception_type, match=r'yt-dlp clipped_audio_download failed with exit code 2'):
            await ytdlp_module._download_audio_as_opus_clipped(
                'https://example.com/watch?v=abc',
                max_duration=timedelta(seconds=15),
                timeout=timedelta(seconds=10),
            )

    error_mock.assert_not_called()
    assert observed['ytdlp_waits'] == 1
    assert observed['ffmpeg_waits'] == 1


@pytest.mark.asyncio
async def test_download_audio_as_opus_clipped_succeeds_when_ytdlp_reports_broken_pipe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {'calls': [], 'ytdlp_waits': 0, 'ffmpeg_waits': 0}

    class _FakeReader:
        def __init__(self, payload: bytes) -> None:
            self._payload = payload

        async def read(self) -> bytes:
            return self._payload

    class _YtDlpProc:
        def __init__(self) -> None:
            self.returncode = 2
            self.stdout = object()
            self.stderr = _FakeReader(b'ERROR: Broken pipe')

        def kill(self) -> None:
            return None

        async def wait(self) -> int:
            observed['ytdlp_waits'] += 1
            return self.returncode

    class _FfmpegProc:
        def __init__(self) -> None:
            self.returncode = 0
            self.stdin = object()
            self.stdout = _FakeReader(b'OggS-clipped')
            self.stderr = _FakeReader(b'')

        def kill(self) -> None:
            return None

        async def wait(self) -> int:
            observed['ffmpeg_waits'] += 1
            return self.returncode

    async def _fake_pipe_stream(source: object, destination: object) -> None:
        return None

    async def _fake_get_selected_audio_codec(url: str, *, timeout: timedelta) -> str | None:
        return 'opus'

    async def _fake_create_subprocess_exec(*args: str, **kwargs: object) -> object:
        observed['calls'].append(args)
        if args[0] == 'yt-dlp':
            return _YtDlpProc()
        return _FfmpegProc()

    monkeypatch.setattr(ytdlp_module, '_pipe_stream', _fake_pipe_stream)
    monkeypatch.setattr(ytdlp_module, '_get_selected_audio_codec', _fake_get_selected_audio_codec)
    monkeypatch.setattr(ytdlp_module.asyncio, 'create_subprocess_exec', _fake_create_subprocess_exec)

    result = await ytdlp_module._download_audio_as_opus_clipped(
        'https://example.com/watch?v=abc',
        max_duration=timedelta(seconds=15),
        timeout=timedelta(seconds=10),
    )

    assert result == b'OggS-clipped'
    assert observed['ytdlp_waits'] == 1
    assert observed['ffmpeg_waits'] == 1
    assert len(observed['calls']) == 2


@pytest.mark.asyncio
async def test_download_audio_as_opus_clipped_raises_on_ffmpeg_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeReader:
        def __init__(self, payload: bytes) -> None:
            self._payload = payload

        async def read(self) -> bytes:
            return self._payload

    class _YtDlpProc:
        def __init__(self) -> None:
            self.returncode = 0
            self.stdout = object()
            self.stderr = _FakeReader(b'')

        def kill(self) -> None:
            return None

        async def wait(self) -> int:
            return self.returncode

    class _FfmpegProc:
        def __init__(self) -> None:
            self.returncode = 1
            self.stdin = object()
            self.stdout = _FakeReader(b'')
            self.stderr = _FakeReader(b'ffmpeg decode failure')
            self.communicate_called = False

        async def communicate(self) -> tuple[bytes, bytes]:
            self.communicate_called = True
            raise AssertionError('ffmpeg communicate must not be called in clipped mode')

        def kill(self) -> None:
            return None

        async def wait(self) -> int:
            return self.returncode

    async def _fake_pipe_stream(source: object, destination: object) -> None:
        return None

    async def _fake_get_selected_audio_codec(url: str, *, timeout: timedelta) -> str | None:
        return 'opus'

    async def _fake_create_subprocess_exec(*args: str, **kwargs: object) -> object:
        if args[0] == 'yt-dlp':
            return _YtDlpProc()
        return _FfmpegProc()

    monkeypatch.setattr(ytdlp_module, '_pipe_stream', _fake_pipe_stream)
    monkeypatch.setattr(ytdlp_module, '_get_selected_audio_codec', _fake_get_selected_audio_codec)
    monkeypatch.setattr(ytdlp_module.asyncio, 'create_subprocess_exec', _fake_create_subprocess_exec)

    with patch.object(ytdlp_module.logger, 'error') as error_mock:
        with pytest.raises(RuntimeError, match=re.escape('ffmpeg clipped_audio_download failed with exit code 1')):
            await ytdlp_module._download_audio_as_opus_clipped(
                'https://example.com/watch?v=abc',
                max_duration=timedelta(seconds=15),
                timeout=timedelta(seconds=10),
            )

    error_mock.assert_not_called()


@pytest.mark.asyncio
async def test_download_audio_as_opus_clipped_never_calls_process_communicate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeReader:
        def __init__(self, payload: bytes) -> None:
            self._payload = payload

        async def read(self) -> bytes:
            return self._payload

    class _YtDlpProc:
        def __init__(self) -> None:
            self.returncode = 0
            self.stdout = object()
            self.stderr = _FakeReader(b'')
            self.communicate_called = False

        async def communicate(self) -> tuple[bytes, bytes]:
            self.communicate_called = True
            raise AssertionError('unexpected ytdlp communicate call')

        def kill(self) -> None:
            return None

        async def wait(self) -> int:
            return self.returncode

    class _FfmpegProc:
        def __init__(self) -> None:
            self.returncode = 0
            self.stdin = object()
            self.stdout = _FakeReader(b'OggS-clipped')
            self.stderr = _FakeReader(b'')
            self.communicate_called = False

        async def communicate(self) -> tuple[bytes, bytes]:
            self.communicate_called = True
            raise AssertionError('unexpected ffmpeg communicate call')

        def kill(self) -> None:
            return None

        async def wait(self) -> int:
            return self.returncode

    ytdlp_proc = _YtDlpProc()
    ffmpeg_proc = _FfmpegProc()

    async def _fake_pipe_stream(source: object, destination: object) -> None:
        return None

    async def _fake_get_selected_audio_codec(url: str, *, timeout: timedelta) -> str | None:
        return 'opus'

    async def _fake_create_subprocess_exec(*args: str, **kwargs: object) -> object:
        if args[0] == 'yt-dlp':
            return ytdlp_proc
        return ffmpeg_proc

    monkeypatch.setattr(ytdlp_module, '_pipe_stream', _fake_pipe_stream)
    monkeypatch.setattr(ytdlp_module, '_get_selected_audio_codec', _fake_get_selected_audio_codec)
    monkeypatch.setattr(ytdlp_module.asyncio, 'create_subprocess_exec', _fake_create_subprocess_exec)

    result = await ytdlp_module._download_audio_as_opus_clipped(
        'https://example.com/watch?v=abc',
        max_duration=timedelta(seconds=15),
        timeout=timedelta(seconds=10),
    )

    assert result == b'OggS-clipped'
    assert ytdlp_proc.communicate_called is False
    assert ffmpeg_proc.communicate_called is False


@pytest.mark.asyncio
async def test_download_audio_as_opus_clipped_timeout_kills_and_waits_for_both_processes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = {
        'ytdlp_terminated': False,
        'ytdlp_killed': False,
        'ffmpeg_terminated': False,
        'ffmpeg_killed': False,
        'ytdlp_waited': False,
        'ffmpeg_waited': False,
    }

    class _FakeReader:
        def __init__(self, payload: bytes = b'') -> None:
            self._payload = payload

        async def read(self) -> bytes:
            await asyncio.sleep(1)
            return self._payload

    class _YtDlpProc:
        def __init__(self) -> None:
            self.returncode = None
            self.stdout = object()
            self.stderr = _FakeReader()

        def terminate(self) -> None:
            observed['ytdlp_terminated'] = True
            self.returncode = 0

        def kill(self) -> None:
            observed['ytdlp_killed'] = True
            self.returncode = -9

        async def wait(self) -> int:
            observed['ytdlp_waited'] = True
            return 0 if self.returncode is None else self.returncode

    class _FfmpegProc:
        def __init__(self) -> None:
            self.returncode = None
            self.stdin = object()
            self.stdout = _FakeReader()
            self.stderr = _FakeReader()

        def terminate(self) -> None:
            observed['ffmpeg_terminated'] = True
            self.returncode = 0

        def kill(self) -> None:
            observed['ffmpeg_killed'] = True
            self.returncode = -9

        async def wait(self) -> int:
            observed['ffmpeg_waited'] = True
            return 0 if self.returncode is None else self.returncode

    async def _fake_pipe_stream(source: object, destination: object) -> None:
        await asyncio.sleep(1)

    async def _fake_get_selected_audio_codec(url: str, *, timeout: timedelta) -> str | None:
        return 'opus'

    async def _fake_create_subprocess_exec(*args: str, **kwargs: object) -> object:
        if args[0] == 'yt-dlp':
            return _YtDlpProc()
        return _FfmpegProc()

    monkeypatch.setattr(ytdlp_module, '_pipe_stream', _fake_pipe_stream)
    monkeypatch.setattr(ytdlp_module, '_get_selected_audio_codec', _fake_get_selected_audio_codec)
    monkeypatch.setattr(ytdlp_module.asyncio, 'create_subprocess_exec', _fake_create_subprocess_exec)

    with pytest.raises(asyncio.TimeoutError):
        await ytdlp_module._download_audio_as_opus_clipped(
            'https://example.com/watch?v=abc',
            max_duration=timedelta(seconds=15),
            timeout=timedelta(milliseconds=1),
        )

    assert observed['ytdlp_terminated'] is True or observed['ytdlp_killed'] is True
    assert observed['ffmpeg_terminated'] is True or observed['ffmpeg_killed'] is True
    assert observed['ytdlp_waited'] is True
    assert observed['ffmpeg_waited'] is True


@pytest.mark.asyncio
async def test_cancelling_clipped_download_reaps_pipeline_before_removing_operation_cookie(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical_cookie_file = tmp_path / 'youtube-cookies.txt'
    canonical_cookie_file.write_bytes(_COOKIE_DATA)
    pipeline_started = asyncio.Event()
    observed: dict[str, object] = {
        'ffmpeg_terminated': False,
        'ffmpeg_killed': False,
        'ffmpeg_waited': False,
        'ytdlp_terminated': False,
        'ytdlp_waited': False,
        'cancelled_pipeline_tasks': 0,
    }

    class _BlockingReader:
        async def read(self) -> bytes:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                observed['cancelled_pipeline_tasks'] = int(observed['cancelled_pipeline_tasks']) + 1
                raise
            return b''

    class _YtDlpProc:
        def __init__(self) -> None:
            self.returncode: int | None = None
            self.stdout = object()
            self.stderr = _BlockingReader()
            self._exited = asyncio.Event()

        def terminate(self) -> None:
            observed['ytdlp_terminated'] = True
            self.returncode = 0
            self._exited.set()

        def kill(self) -> None:
            self.returncode = -9
            self._exited.set()

        async def wait(self) -> int:
            await self._exited.wait()
            operation_cookie_file = observed['operation_cookie_file']
            assert isinstance(operation_cookie_file, Path)
            assert operation_cookie_file.is_file()
            observed['ytdlp_waited'] = True
            return self.returncode or 0

    class _FfmpegProc:
        def __init__(self) -> None:
            self.returncode: int | None = None
            self.stdin = object()
            self.stdout = _BlockingReader()
            self.stderr = _BlockingReader()
            self._exited = asyncio.Event()

        def terminate(self) -> None:
            observed['ffmpeg_terminated'] = True

        def kill(self) -> None:
            observed['ffmpeg_killed'] = True
            self.returncode = -9
            self._exited.set()

        async def wait(self) -> int:
            pipeline_started.set()
            await self._exited.wait()
            operation_cookie_file = observed['operation_cookie_file']
            assert isinstance(operation_cookie_file, Path)
            assert operation_cookie_file.is_file()
            observed['ffmpeg_waited'] = True
            return self.returncode or -9

    async def _fake_pipe_stream(source: object, destination: object) -> None:
        del source, destination
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            observed['cancelled_pipeline_tasks'] = int(observed['cancelled_pipeline_tasks']) + 1
            raise

    async def _fake_get_media_duration(url: str, *, timeout: timedelta) -> timedelta | None:
        return timedelta(seconds=31)

    async def _fake_get_selected_audio_codec(url: str, *, timeout: timedelta) -> str | None:
        return 'opus'

    async def _fake_create_subprocess_exec(*args: str, **kwargs: object) -> object:
        if args[0] == 'yt-dlp':
            observed['operation_cookie_file'] = Path(args[args.index('--cookies') + 1])
            return _YtDlpProc()
        return _FfmpegProc()

    monkeypatch.setattr(ytdlp_module, '_PROCESS_TERMINATION_GRACE_SECONDS', 0.01)
    monkeypatch.setattr(ytdlp_module, '_get_media_duration', _fake_get_media_duration)
    monkeypatch.setattr(ytdlp_module, '_get_selected_audio_codec', _fake_get_selected_audio_codec)
    monkeypatch.setattr(ytdlp_module, '_pipe_stream', _fake_pipe_stream)
    monkeypatch.setattr(ytdlp_module.asyncio, 'create_subprocess_exec', _fake_create_subprocess_exec)
    operation = asyncio.create_task(
        ytdlp_module.download_audio_as_opus(
            'https://example.com/watch?v=abc',
            cookie_file=canonical_cookie_file,
            max_duration=timedelta(seconds=30),
        ),
    )
    await pipeline_started.wait()
    operation.cancel()

    with patch.object(ytdlp_module.logger, 'error') as error_mock:
        with pytest.raises(asyncio.CancelledError):
            await operation

    operation_cookie_file = observed['operation_cookie_file']
    assert isinstance(operation_cookie_file, Path)
    assert observed['ffmpeg_terminated'] is True
    assert observed['ffmpeg_killed'] is True
    assert observed['ffmpeg_waited'] is True
    assert observed['ytdlp_terminated'] is True
    assert observed['ytdlp_waited'] is True
    assert observed['cancelled_pipeline_tasks'] == 4
    assert not operation_cookie_file.exists()
    error_mock.assert_not_called()


@pytest.mark.asyncio
async def test_download_audio_as_opus_clipped_success_terminates_hanging_ytdlp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = {
        'ytdlp_terminated': False,
        'ytdlp_killed': False,
    }

    class _FakeReader:
        def __init__(self, payload: bytes = b'') -> None:
            self._payload = payload

        async def read(self) -> bytes:
            return self._payload

    class _YtDlpProc:
        def __init__(self) -> None:
            self.returncode = None
            self.stdout = object()
            self.stderr = _FakeReader()

        def terminate(self) -> None:
            observed['ytdlp_terminated'] = True
            self.returncode = 0

        def kill(self) -> None:
            observed['ytdlp_killed'] = True
            self.returncode = -9

        async def wait(self) -> int:
            while self.returncode is None:
                await asyncio.sleep(0.01)
            return self.returncode

    class _FfmpegProc:
        def __init__(self) -> None:
            self.returncode = 0
            self.stdin = object()
            self.stdout = _FakeReader(b'OggS-clipped')
            self.stderr = _FakeReader(b'')

        async def wait(self) -> int:
            return self.returncode

    async def _fake_pipe_stream(source: object, destination: object) -> None:
        return None

    async def _fake_get_selected_audio_codec(url: str, *, timeout: timedelta) -> str | None:
        return 'opus'

    async def _fake_create_subprocess_exec(*args: str, **kwargs: object) -> object:
        if args[0] == 'yt-dlp':
            return _YtDlpProc()
        return _FfmpegProc()

    monkeypatch.setattr(ytdlp_module, '_pipe_stream', _fake_pipe_stream)
    monkeypatch.setattr(ytdlp_module, '_get_selected_audio_codec', _fake_get_selected_audio_codec)
    monkeypatch.setattr(ytdlp_module.asyncio, 'create_subprocess_exec', _fake_create_subprocess_exec)

    result = await ytdlp_module._download_audio_as_opus_clipped(
        'https://example.com/watch?v=abc',
        max_duration=timedelta(seconds=15),
        timeout=timedelta(seconds=2),
    )

    assert result.startswith(b'OggS')
    assert observed['ytdlp_terminated'] is True
    assert observed['ytdlp_killed'] is False


@pytest.mark.asyncio
async def test_download_audio_as_opus_clipped_timeout_cleanup_ignores_process_lookup_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeReader:
        async def read(self) -> bytes:
            await asyncio.sleep(1)
            return b''

    class _YtDlpProc:
        def __init__(self) -> None:
            self.returncode = None
            self.stdout = object()
            self.stderr = _FakeReader()

        def terminate(self) -> None:
            raise ProcessLookupError

        def kill(self) -> None:
            raise ProcessLookupError

        async def wait(self) -> int:
            self.returncode = 0
            return self.returncode

    class _FfmpegProc:
        def __init__(self) -> None:
            self.returncode = None
            self.stdin = object()
            self.stdout = _FakeReader()
            self.stderr = _FakeReader()

        def terminate(self) -> None:
            raise ProcessLookupError

        def kill(self) -> None:
            raise ProcessLookupError

        async def wait(self) -> int:
            self.returncode = 0
            return self.returncode

    async def _fake_pipe_stream(source: object, destination: object) -> None:
        await asyncio.sleep(1)

    async def _fake_get_selected_audio_codec(url: str, *, timeout: timedelta) -> str | None:
        return 'opus'

    async def _fake_create_subprocess_exec(*args: str, **kwargs: object) -> object:
        if args[0] == 'yt-dlp':
            return _YtDlpProc()
        return _FfmpegProc()

    monkeypatch.setattr(ytdlp_module, '_pipe_stream', _fake_pipe_stream)
    monkeypatch.setattr(ytdlp_module, '_get_selected_audio_codec', _fake_get_selected_audio_codec)
    monkeypatch.setattr(ytdlp_module.asyncio, 'create_subprocess_exec', _fake_create_subprocess_exec)

    with pytest.raises(asyncio.TimeoutError):
        await ytdlp_module._download_audio_as_opus_clipped(
            'https://example.com/watch?v=abc',
            max_duration=timedelta(seconds=15),
            timeout=timedelta(milliseconds=1),
        )


@pytest.mark.asyncio
async def test_pipe_stream_tolerates_broken_pipe_on_wait_closed() -> None:
    class _FakeSource:
        def __init__(self) -> None:
            self._reads = [b'data', b'']

        async def read(self, n: int) -> bytes:
            return self._reads.pop(0)

    class _FakeDestination:
        def __init__(self) -> None:
            self.writes: list[bytes] = []
            self.closed = False

        def write(self, chunk: bytes) -> None:
            self.writes.append(chunk)

        async def drain(self) -> None:
            return None

        def close(self) -> None:
            self.closed = True

        async def wait_closed(self) -> None:
            raise BrokenPipeError

    source = _FakeSource()
    destination = _FakeDestination()

    await ytdlp_module._pipe_stream(source, destination)

    assert destination.writes == [b'data']
    assert destination.closed is True


@pytest.mark.asyncio
async def test_pipe_stream_tolerates_connection_reset_on_wait_closed() -> None:
    class _FakeSource:
        def __init__(self) -> None:
            self._reads = [b'data', b'']

        async def read(self, n: int) -> bytes:
            return self._reads.pop(0)

    class _FakeDestination:
        def __init__(self) -> None:
            self.writes: list[bytes] = []
            self.closed = False

        def write(self, chunk: bytes) -> None:
            self.writes.append(chunk)

        async def drain(self) -> None:
            return None

        def close(self) -> None:
            self.closed = True

        async def wait_closed(self) -> None:
            raise ConnectionResetError

    source = _FakeSource()
    destination = _FakeDestination()

    await ytdlp_module._pipe_stream(source, destination)

    assert destination.writes == [b'data']
    assert destination.closed is True


@pytest.mark.asyncio
async def test_download_audio_as_opus_clipped_succeeds_when_pipe_close_hits_broken_pipe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeReader:
        def __init__(self, chunks: list[bytes] | bytes) -> None:
            if isinstance(chunks, bytes):
                self._chunks = [chunks]
            else:
                self._chunks = list(chunks)

        async def read(self, n: int = -1) -> bytes:
            if not self._chunks:
                return b''
            return self._chunks.pop(0)

    class _FakeWriter:
        def __init__(self) -> None:
            self.closed = False
            self.writes: list[bytes] = []

        def write(self, chunk: bytes) -> None:
            self.writes.append(chunk)

        async def drain(self) -> None:
            return None

        def close(self) -> None:
            self.closed = True

        async def wait_closed(self) -> None:
            raise BrokenPipeError

    class _YtDlpProc:
        def __init__(self) -> None:
            self.returncode = 0
            self.stdout = _FakeReader([b'opus-bytes', b''])
            self.stderr = _FakeReader(b'')

        def kill(self) -> None:
            return None

        async def wait(self) -> int:
            return self.returncode

    class _FfmpegProc:
        def __init__(self) -> None:
            self.returncode = 0
            self.stdin = _FakeWriter()
            self.stdout = _FakeReader(b'OggS-clipped')
            self.stderr = _FakeReader(b'')

        def kill(self) -> None:
            return None

        async def wait(self) -> int:
            return self.returncode

    async def _fake_create_subprocess_exec(*args: str, **kwargs: object) -> object:
        if args[0] == 'yt-dlp':
            return _YtDlpProc()
        return _FfmpegProc()

    async def _fake_get_selected_audio_codec(url: str, *, timeout: timedelta) -> str | None:
        return 'opus'

    monkeypatch.setattr(ytdlp_module, '_get_selected_audio_codec', _fake_get_selected_audio_codec)
    monkeypatch.setattr(ytdlp_module.asyncio, 'create_subprocess_exec', _fake_create_subprocess_exec)

    result = await ytdlp_module._download_audio_as_opus_clipped(
        'https://example.com/watch?v=abc',
        max_duration=timedelta(seconds=15),
        timeout=timedelta(seconds=10),
    )

    assert result == b'OggS-clipped'


@pytest.mark.asyncio
async def test_download_audio_as_opus_clipped_uses_libopus_when_source_codec_is_not_opus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, tuple[str, ...]] = {}

    class _FakeReader:
        def __init__(self, payload: bytes) -> None:
            self._payload = payload

        async def read(self) -> bytes:
            return self._payload

    class _YtDlpProc:
        def __init__(self) -> None:
            self.returncode = 0
            self.stdout = object()
            self.stderr = _FakeReader(b'')

        def kill(self) -> None:
            return None

        async def wait(self) -> int:
            return self.returncode

    class _FfmpegProc:
        def __init__(self) -> None:
            self.returncode = 0
            self.stdin = object()
            self.stdout = _FakeReader(b'OggS-clipped')
            self.stderr = _FakeReader(b'')

        def kill(self) -> None:
            return None

        async def wait(self) -> int:
            return self.returncode

    async def _fake_pipe_stream(source: object, destination: object) -> None:
        return None

    async def _fake_get_selected_audio_codec(url: str, *, timeout: timedelta) -> str | None:
        return 'mp4a.40.2'

    async def _fake_create_subprocess_exec(*args: str, **kwargs: object) -> object:
        if args[0] == 'yt-dlp':
            observed['yt-dlp'] = args
            return _YtDlpProc()
        observed['ffmpeg'] = args
        return _FfmpegProc()

    monkeypatch.setattr(ytdlp_module, '_pipe_stream', _fake_pipe_stream)
    monkeypatch.setattr(ytdlp_module, '_get_selected_audio_codec', _fake_get_selected_audio_codec)
    monkeypatch.setattr(ytdlp_module.asyncio, 'create_subprocess_exec', _fake_create_subprocess_exec)

    result = await ytdlp_module._download_audio_as_opus_clipped(
        'https://example.com/watch?v=abc',
        max_duration=timedelta(seconds=15),
        timeout=timedelta(seconds=10),
    )

    assert result == b'OggS-clipped'
    ffmpeg_args = observed['ffmpeg']
    assert ffmpeg_args[ffmpeg_args.index('-c:a') + 1] == 'libopus'
    assert '-b:a' in ffmpeg_args
    assert ffmpeg_args[ffmpeg_args.index('-b:a') + 1] == '160k'
    assert '-vbr' in ffmpeg_args
    assert ffmpeg_args[ffmpeg_args.index('-vbr') + 1] == 'on'
    assert '-compression_level' in ffmpeg_args
    assert ffmpeg_args[ffmpeg_args.index('-compression_level') + 1] == '10'


@pytest.mark.asyncio
async def test_download_audio_as_opus_clipped_uses_libopus_when_source_codec_is_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, tuple[str, ...]] = {}

    class _FakeReader:
        def __init__(self, payload: bytes) -> None:
            self._payload = payload

        async def read(self) -> bytes:
            return self._payload

    class _YtDlpProc:
        def __init__(self) -> None:
            self.returncode = 0
            self.stdout = object()
            self.stderr = _FakeReader(b'')

        def kill(self) -> None:
            return None

        async def wait(self) -> int:
            return self.returncode

    class _FfmpegProc:
        def __init__(self) -> None:
            self.returncode = 0
            self.stdin = object()
            self.stdout = _FakeReader(b'OggS-clipped')
            self.stderr = _FakeReader(b'')

        def kill(self) -> None:
            return None

        async def wait(self) -> int:
            return self.returncode

    async def _fake_pipe_stream(source: object, destination: object) -> None:
        return None

    async def _fake_get_selected_audio_codec(url: str, *, timeout: timedelta) -> str | None:
        return 'NA'

    async def _fake_create_subprocess_exec(*args: str, **kwargs: object) -> object:
        if args[0] == 'yt-dlp':
            observed['yt-dlp'] = args
            return _YtDlpProc()
        observed['ffmpeg'] = args
        return _FfmpegProc()

    monkeypatch.setattr(ytdlp_module, '_pipe_stream', _fake_pipe_stream)
    monkeypatch.setattr(ytdlp_module, '_get_selected_audio_codec', _fake_get_selected_audio_codec)
    monkeypatch.setattr(ytdlp_module.asyncio, 'create_subprocess_exec', _fake_create_subprocess_exec)

    result = await ytdlp_module._download_audio_as_opus_clipped(
        'https://example.com/watch?v=abc',
        max_duration=timedelta(seconds=15),
        timeout=timedelta(seconds=10),
    )

    assert result == b'OggS-clipped'
    ffmpeg_args = observed['ffmpeg']
    assert ffmpeg_args[ffmpeg_args.index('-c:a') + 1] == 'libopus'


@pytest.mark.asyncio
async def test_download_audio_as_opus_clipped_raises_timeout_when_probe_consumes_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {'create_subprocess_called': False}

    class _FakeLoop:
        def __init__(self) -> None:
            self._calls = 0

        def time(self) -> float:
            self._calls += 1
            if self._calls == 1:
                return 100.0
            return 111.0

    async def _fake_get_selected_audio_codec(url: str, *, timeout: timedelta) -> str | None:
        return 'opus'

    async def _fake_create_subprocess_exec(*args: str, **kwargs: object) -> object:
        observed['create_subprocess_called'] = True
        raise AssertionError('create_subprocess_exec should not be called when timeout is exhausted')

    monkeypatch.setattr(ytdlp_module.asyncio, 'get_running_loop', lambda: _FakeLoop())
    monkeypatch.setattr(ytdlp_module, '_get_selected_audio_codec', _fake_get_selected_audio_codec)
    monkeypatch.setattr(ytdlp_module.asyncio, 'create_subprocess_exec', _fake_create_subprocess_exec)

    with pytest.raises(asyncio.TimeoutError):
        await ytdlp_module._download_audio_as_opus_clipped(
            'https://example.com/watch?v=abc',
            max_duration=timedelta(seconds=15),
            timeout=timedelta(seconds=10),
        )

    assert observed['create_subprocess_called'] is False
