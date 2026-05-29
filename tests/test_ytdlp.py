import asyncio
import re
import urllib.error
from datetime import timedelta
from pathlib import Path

import pytest

from timeline_hub.infra import ytdlp as ytdlp_module
from timeline_hub.infra.ytdlp import YtDlpMetadataError


@pytest.mark.asyncio
async def test_fetch_track_info_with_cover_and_metadata_returns_all_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

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
    assert '-f' in args
    assert args[args.index('-f') + 1] == 'bestaudio'
    assert '--extract-audio' in args
    assert '--audio-format' in args
    assert 'opus' in args
    assert '--quiet' in args
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

    with pytest.raises(RuntimeError, match=re.escape('yt-dlp failed: failure details')):
        await ytdlp_module.download_audio_as_opus('https://example.com/watch?v=abc')


@pytest.mark.asyncio
async def test_download_audio_as_opus_kills_and_waits_on_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = {'killed': False, 'waited': False}

    class _FakeProc:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return b'', b''

        def kill(self) -> None:
            observed['killed'] = True

        async def wait(self) -> int:
            observed['waited'] = True
            return self.returncode

    async def _fake_create_subprocess_exec(*args: str, **kwargs: object) -> _FakeProc:
        return _FakeProc()

    async def _fake_wait_for(awaitable: object, timeout: float) -> tuple[bytes, bytes]:
        close = getattr(awaitable, 'close', None)
        if callable(close):
            close()
        raise asyncio.TimeoutError

    monkeypatch.setattr(ytdlp_module.asyncio, 'create_subprocess_exec', _fake_create_subprocess_exec)
    monkeypatch.setattr(ytdlp_module.asyncio, 'wait_for', _fake_wait_for)

    with pytest.raises(asyncio.TimeoutError):
        await ytdlp_module.download_audio_as_opus('https://example.com/watch?v=abc')

    assert observed['killed'] is True
    assert observed['waited'] is True


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
    assert '--print' in args
    assert '%(duration)s' in args
    assert '--skip-download' in args
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

    with pytest.raises(RuntimeError, match=re.escape('yt-dlp failed: probe error')):
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

    monkeypatch.setattr(ytdlp_module, 'get_media_duration', _fake_get_media_duration)
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

    monkeypatch.setattr(ytdlp_module, 'get_media_duration', _fake_get_media_duration)
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

    monkeypatch.setattr(ytdlp_module, 'get_media_duration', _fake_get_media_duration)
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

    monkeypatch.setattr(ytdlp_module, 'get_media_duration', _fake_get_media_duration)
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

    monkeypatch.setattr(ytdlp_module, 'get_media_duration', _fake_get_media_duration)
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

    monkeypatch.setattr(ytdlp_module, 'get_media_duration', _fake_get_media_duration)
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

    monkeypatch.setattr(ytdlp_module, 'get_media_duration', _fake_get_media_duration)
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

    monkeypatch.setattr(ytdlp_module, 'get_media_duration', _unexpected_get_media_duration)
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
async def test_download_audio_as_opus_clipped_builds_pipeline_and_returns_bytes(
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
    ytdlp_args = observed['yt-dlp']
    ffmpeg_args = observed['ffmpeg']
    assert '-o' in ytdlp_args
    assert ytdlp_args[ytdlp_args.index('-o') + 1] == '-'
    assert '-f' in ytdlp_args
    assert ytdlp_args[ytdlp_args.index('-f') + 1] == 'bestaudio'
    assert 'pipe:0' in ffmpeg_args
    assert '-t' in ffmpeg_args
    assert ffmpeg_args[ffmpeg_args.index('-t') + 1] == '15.0'
    assert '-c:a' in ffmpeg_args
    assert ffmpeg_args[ffmpeg_args.index('-c:a') + 1] == 'copy'
    assert '-f' in ffmpeg_args
    assert ffmpeg_args[ffmpeg_args.index('-f') + 1] == 'opus'
    assert ffmpeg_args[-1] == 'pipe:1'


@pytest.mark.asyncio
async def test_download_audio_as_opus_clipped_tolerates_ytdlp_broken_pipe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
        timeout=timedelta(seconds=10),
    )
    assert result == b'OggS-clipped'


@pytest.mark.asyncio
async def test_download_audio_as_opus_clipped_returns_success_when_ffmpeg_succeeds_even_if_ytdlp_nonzero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeReader:
        def __init__(self, payload: bytes) -> None:
            self._payload = payload

        async def read(self) -> bytes:
            return self._payload

    class _YtDlpProc:
        def __init__(self) -> None:
            self.returncode = 2
            self.stdout = object()
            self.stderr = _FakeReader(b'network error')

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
        timeout=timedelta(seconds=10),
    )
    assert result == b'OggS-clipped'


@pytest.mark.asyncio
async def test_download_audio_as_opus_clipped_prefers_ytdlp_error_when_both_processes_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeReader:
        def __init__(self, payload: bytes) -> None:
            self._payload = payload

        async def read(self) -> bytes:
            return self._payload

    class _YtDlpProc:
        def __init__(self) -> None:
            self.returncode = 2
            self.stdout = object()
            self.stderr = _FakeReader(b'ytdlp boom')

        def kill(self) -> None:
            return None

        async def wait(self) -> int:
            return self.returncode

    class _FfmpegProc:
        def __init__(self) -> None:
            self.returncode = 183
            self.stdin = object()
            self.stdout = _FakeReader(b'')
            self.stderr = _FakeReader(b'Invalid data found when processing input')

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

    with pytest.raises(RuntimeError, match=re.escape('yt-dlp failed: ytdlp boom')):
        await ytdlp_module._download_audio_as_opus_clipped(
            'https://example.com/watch?v=abc',
            max_duration=timedelta(seconds=15),
            timeout=timedelta(seconds=10),
        )


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

    with pytest.raises(RuntimeError, match=re.escape('ffmpeg failed: ffmpeg decode failure')):
        await ytdlp_module._download_audio_as_opus_clipped(
            'https://example.com/watch?v=abc',
            max_duration=timedelta(seconds=15),
            timeout=timedelta(seconds=10),
        )


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
async def test_download_audio_as_opus_clipped_propagates_timeout_to_codec_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

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
        observed['probe_timeout'] = timeout
        return 'opus'

    async def _fake_create_subprocess_exec(*args: str, **kwargs: object) -> object:
        if args[0] == 'yt-dlp':
            return _YtDlpProc()
        return _FfmpegProc()

    timeout = timedelta(seconds=10)
    monkeypatch.setattr(ytdlp_module, '_pipe_stream', _fake_pipe_stream)
    monkeypatch.setattr(ytdlp_module, '_get_selected_audio_codec', _fake_get_selected_audio_codec)
    monkeypatch.setattr(ytdlp_module.asyncio, 'create_subprocess_exec', _fake_create_subprocess_exec)

    result = await ytdlp_module._download_audio_as_opus_clipped(
        'https://example.com/watch?v=abc',
        max_duration=timedelta(seconds=15),
        timeout=timeout,
    )

    assert result == b'OggS-clipped'
    assert observed['probe_timeout'] == timeout


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
