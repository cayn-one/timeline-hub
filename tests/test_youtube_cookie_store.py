import asyncio
import inspect
import os
import stat
from pathlib import Path

import pytest

import timeline_hub.services.youtube_cookies as cookie_store_module
from timeline_hub.services.youtube_cookies import (
    YoutubeCookieMaterializationError,
    YoutubeCookieNotMaterializedError,
    YoutubeCookieStore,
    YoutubeCookieStoreError,
    YoutubeCookieValidationError,
)

_VALID_COOKIE_FILE = b'# Netscape HTTP Cookie File\n.youtube.com\tTRUE\t/\tTRUE\t0\tSID\tvalue\n'


class _FakeS3Client:
    def __init__(self, data: bytes = _VALID_COOKIE_FILE) -> None:
        self.data = data
        self.get_calls = 0
        self.put_calls: list[tuple[str, bytes, str | None]] = []

    async def get_bytes(self, key: str) -> bytes:
        assert key == 'youtube-cookies/cookies.txt'
        self.get_calls += 1
        return self.data

    async def put_bytes(self, key: str, data: bytes, *, content_type: str | None = None) -> None:
        self.put_calls.append((key, data, content_type))


def _store(tmp_path: Path, s3_client: _FakeS3Client) -> YoutubeCookieStore:
    return YoutubeCookieStore(s3_client, runtime_directory=tmp_path / 'runtime')


@pytest.mark.asyncio
async def test_constructor_owns_runtime_directory_and_fixed_filename(tmp_path: Path) -> None:
    store = _store(tmp_path, _FakeS3Client())
    signature = inspect.signature(YoutubeCookieStore)
    snapshot = await store.refresh()

    assert signature.parameters['runtime_directory'].default == Path('/tmp/timeline-hub')
    assert 'local_path' not in signature.parameters
    assert snapshot.path == tmp_path / 'runtime' / 'youtube-cookies.txt'


@pytest.mark.parametrize('runtime_directory', [Path('/'), Path('/tmp'), Path('/var/tmp'), Path('/tmp/child/..')])
def test_constructor_rejects_unsafe_shared_runtime_directories(runtime_directory: Path) -> None:
    with pytest.raises(YoutubeCookieMaterializationError, match='unsafe'):
        YoutubeCookieStore(_FakeS3Client(), runtime_directory=runtime_directory)


@pytest.mark.asyncio
async def test_upload_does_not_depend_on_runtime_directory_state(tmp_path: Path) -> None:
    runtime_path = tmp_path / 'runtime'
    runtime_path.write_text('not a directory')
    source = tmp_path / 'cookies.txt'
    source.write_bytes(_VALID_COOKIE_FILE)
    s3_client = _FakeS3Client()
    store = YoutubeCookieStore(s3_client, runtime_directory=runtime_path)

    uploaded_bytes = await store.upload(source)

    assert uploaded_bytes == len(_VALID_COOKIE_FILE)
    assert s3_client.put_calls == [('youtube-cookies/cookies.txt', _VALID_COOKIE_FILE, 'text/plain')]
    with pytest.raises(YoutubeCookieMaterializationError, match='not a directory'):
        await store.refresh()


def test_current_requires_successful_materialization(tmp_path: Path) -> None:
    store = _store(tmp_path, _FakeS3Client())

    with pytest.raises(YoutubeCookieNotMaterializedError):
        store.current()


@pytest.mark.asyncio
async def test_refresh_validates_and_materializes_restrictive_cookie_file(tmp_path: Path) -> None:
    store = _store(tmp_path, _FakeS3Client())

    snapshot = await store.refresh()

    assert snapshot.path.read_bytes() == _VALID_COOKIE_FILE
    assert stat.S_IMODE(snapshot.path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(snapshot.path.stat().st_mode) == 0o600
    assert store.current() == snapshot


@pytest.mark.asyncio
async def test_current_rejects_externally_deleted_cookie_file(tmp_path: Path) -> None:
    store = _store(tmp_path, _FakeS3Client())
    snapshot = await store.refresh()
    snapshot.path.unlink()

    with pytest.raises(YoutubeCookieNotMaterializedError):
        store.current()


@pytest.mark.asyncio
async def test_materialization_only_chmods_declared_runtime_directory(tmp_path: Path) -> None:
    parent = tmp_path / 'parent'
    parent.mkdir(mode=0o755)
    os.chmod(parent, 0o755)
    runtime_directory = parent / 'runtime'
    store = YoutubeCookieStore(_FakeS3Client(), runtime_directory=runtime_directory)

    snapshot = await store.refresh()

    assert stat.S_IMODE(parent.stat().st_mode) == 0o755
    assert stat.S_IMODE(runtime_directory.stat().st_mode) == 0o700
    assert stat.S_IMODE(snapshot.path.stat().st_mode) == 0o600


@pytest.mark.parametrize(
    'data',
    [
        b'',
        b'\xff',
        b'# not netscape\n.youtube.com\tTRUE\t/\tTRUE\t0\tSID\tvalue\n',
        b'# Netscape HTTP Cookie File\n# comment only\n',
        b'# Netscape HTTP Cookie File\n.youtube.com\tTRUE\t/\tTRUE\t0\tSID\n',
    ],
)
@pytest.mark.asyncio
async def test_refresh_rejects_malformed_netscape_cookie_files(tmp_path: Path, data: bytes) -> None:
    store = _store(tmp_path, _FakeS3Client(data))

    with pytest.raises(YoutubeCookieValidationError):
        await store.refresh()


@pytest.mark.asyncio
async def test_refresh_accepts_httponly_netscape_records(tmp_path: Path) -> None:
    store = _store(
        tmp_path,
        _FakeS3Client(b'# Netscape HTTP Cookie File\n#HttpOnly_.youtube.com\tTRUE\t/\tTRUE\t0\tSID\tvalue\n'),
    )

    snapshot = await store.refresh()

    assert snapshot.path.is_file()


@pytest.mark.asyncio
async def test_local_materialization_failures_preserve_previous_cookie_and_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    materialization_failure_stage: str,
) -> None:
    s3_client = _FakeS3Client()
    store = _store(tmp_path, s3_client)
    first = await store.refresh()
    previous_data = first.path.read_bytes()
    s3_client.data = _VALID_COOKIE_FILE.replace(b'value', b'new-value')

    failure = OSError(f'{materialization_failure_stage} failed')
    if materialization_failure_stage == 'write':
        original_named_temporary_file = cookie_store_module.tempfile.NamedTemporaryFile

        class _WriteFailingTemporaryFile:
            def __init__(self, *args: object, **kwargs: object) -> None:
                self._file = original_named_temporary_file(*args, **kwargs)
                self.name = self._file.name

            def __enter__(self) -> _WriteFailingTemporaryFile:
                self._file.__enter__()
                return self

            def __exit__(self, exc_type, exc, traceback) -> bool | None:
                return self._file.__exit__(exc_type, exc, traceback)

            def write(self, data: bytes) -> int:
                raise failure

            def flush(self) -> None:
                self._file.flush()

            def fileno(self) -> int:
                return self._file.fileno()

        monkeypatch.setattr(cookie_store_module.tempfile, 'NamedTemporaryFile', _WriteFailingTemporaryFile)
    elif materialization_failure_stage == 'fsync':
        monkeypatch.setattr(cookie_store_module.os, 'fsync', lambda file_descriptor: (_ for _ in ()).throw(failure))
    elif materialization_failure_stage == 'chmod':
        original_chmod = cookie_store_module.os.chmod

        def fail_temporary_chmod(path: Path, mode: int) -> None:
            if Path(path).name.startswith('.youtube-cookies.txt.'):
                raise failure
            original_chmod(path, mode)

        monkeypatch.setattr(cookie_store_module.os, 'chmod', fail_temporary_chmod)
    else:
        monkeypatch.setattr(
            cookie_store_module.os,
            'replace',
            lambda source, destination: (_ for _ in ()).throw(failure),
        )

    with pytest.raises(YoutubeCookieMaterializationError) as exc_info:
        await store.refresh()

    assert exc_info.value.__cause__ is failure
    assert first.path.read_bytes() == previous_data
    assert store.current() == first
    assert not list(first.path.parent.glob('.youtube-cookies.txt.*'))


@pytest.fixture(params=['write', 'fsync', 'chmod', 'replace'])
def materialization_failure_stage(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.mark.asyncio
async def test_refresh_after_rejection_coalesces_a_newer_snapshot(tmp_path: Path) -> None:
    s3_client = _FakeS3Client()
    store = _store(tmp_path, s3_client)
    rejected = await store.refresh()
    refreshed = await store.refresh()

    result = await store.refresh_after_rejection(rejected)

    assert result == refreshed
    assert s3_client.get_calls == 2


@pytest.mark.asyncio
async def test_refresh_after_rejection_rejects_snapshot_from_another_store(tmp_path: Path) -> None:
    runtime_directory = tmp_path / 'runtime'
    first_store = YoutubeCookieStore(_FakeS3Client(), runtime_directory=runtime_directory)
    second_store = YoutubeCookieStore(_FakeS3Client(), runtime_directory=runtime_directory)
    await first_store.refresh()
    foreign_snapshot = await second_store.refresh()

    with pytest.raises(ValueError, match='belongs to another store'):
        await first_store.refresh_after_rejection(foreign_snapshot)


@pytest.mark.asyncio
async def test_stale_snapshot_with_deleted_current_file_performs_real_refresh(tmp_path: Path) -> None:
    s3_client = _FakeS3Client()
    store = _store(tmp_path, s3_client)
    rejected = await store.refresh()
    newer = await store.refresh()
    newer.path.unlink()

    refreshed = await store.refresh_after_rejection(rejected)

    assert refreshed.path.is_file()
    assert refreshed != newer
    assert s3_client.get_calls == 3


@pytest.mark.asyncio
async def test_concurrent_rejection_refreshes_fetch_once(tmp_path: Path) -> None:
    s3_client = _FakeS3Client()
    store = _store(tmp_path, s3_client)
    rejected = await store.refresh()
    s3_client.get_calls = 0

    first, second = await asyncio.gather(
        store.refresh_after_rejection(rejected),
        store.refresh_after_rejection(rejected),
    )

    assert first == second
    assert s3_client.get_calls == 1


@pytest.mark.asyncio
async def test_cancelled_refresh_leaves_no_temporary_cookie_file(tmp_path: Path) -> None:
    class _CancelledS3Client(_FakeS3Client):
        async def get_bytes(self, key: str) -> bytes:
            raise asyncio.CancelledError

    store = _store(tmp_path, _CancelledS3Client())

    with pytest.raises(asyncio.CancelledError):
        await store.refresh()

    assert not (tmp_path / 'runtime').exists()


@pytest.mark.asyncio
async def test_upload_validates_source_and_uses_fixed_key(tmp_path: Path) -> None:
    source = tmp_path / 'cookies.txt'
    source.write_bytes(_VALID_COOKIE_FILE)
    s3_client = _FakeS3Client()
    store = _store(tmp_path, s3_client)

    uploaded_bytes = await store.upload(source)

    assert uploaded_bytes == len(_VALID_COOKIE_FILE)
    assert s3_client.put_calls == [('youtube-cookies/cookies.txt', _VALID_COOKIE_FILE, 'text/plain')]


@pytest.mark.asyncio
async def test_upload_rejects_invalid_source_before_s3_write(tmp_path: Path) -> None:
    source = tmp_path / 'cookies.txt'
    source.write_bytes(b'not cookies')
    s3_client = _FakeS3Client()
    store = _store(tmp_path, s3_client)

    with pytest.raises(YoutubeCookieValidationError):
        await store.upload(source)

    assert not s3_client.put_calls


@pytest.mark.asyncio
async def test_upload_wraps_missing_source_without_exposing_contents(tmp_path: Path) -> None:
    store = _store(tmp_path, _FakeS3Client())

    with pytest.raises(YoutubeCookieStoreError, match='failed to read'):
        await store.upload(tmp_path / 'missing.txt')
