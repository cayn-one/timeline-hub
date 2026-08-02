import asyncio
import contextlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from async_s3 import Key, S3Client, S3ContentType

_COOKIE_KEY: Key = S3Client.join('youtube-cookies', 'cookies.txt')
_LOCAL_FILENAME = 'youtube-cookies.txt'
_DEFAULT_RUNTIME_DIRECTORY = Path('/tmp/timeline-hub')
_UNSAFE_RUNTIME_DIRECTORIES = frozenset({Path('/'), Path('/tmp'), Path('/var/tmp')})
_NETSCAPE_HEADER = '# Netscape HTTP Cookie File'


class YoutubeCookieStoreError(RuntimeError):
    """Raised when the YouTube cookie artifact cannot be safely used."""


class YoutubeCookieNotMaterializedError(YoutubeCookieStoreError):
    """Raised before the store has successfully materialized a cookie file."""


class YoutubeCookieValidationError(YoutubeCookieStoreError):
    """Raised when a cookie file is not structurally valid Netscape data."""


class YoutubeCookieMaterializationError(YoutubeCookieStoreError):
    """Raised when the cookie file cannot be safely materialized locally."""


@dataclass(frozen=True, slots=True)
class YoutubeCookieSnapshot:
    """Generation-tagged reference to the store's current local cookie path.

    The snapshot object is immutable, but a later refresh atomically replaces the
    file at `path`.
    """

    path: Path
    _generation: int
    _store_identity: object


class YoutubeCookieStore:
    """Own the canonical YouTube cookie artifact and its local materialization."""

    def __init__(
        self,
        s3_client: S3Client,
        *,
        runtime_directory: Path = _DEFAULT_RUNTIME_DIRECTORY,
    ) -> None:
        runtime_directory = _normalize_runtime_directory(runtime_directory)
        self._s3_client = s3_client
        self._runtime_directory = runtime_directory
        self._lock = asyncio.Lock()
        self._generation = 0
        self._store_identity = object()
        self._snapshot: YoutubeCookieSnapshot | None = None

    def current(self) -> YoutubeCookieSnapshot:
        """Return the currently materialized cookie file.

        Raises:
            YoutubeCookieNotMaterializedError: If no successful refresh has materialized a file.
        """
        if self._snapshot is None or not self._snapshot.path.is_file():
            raise YoutubeCookieNotMaterializedError('YouTube cookie file is not materialized')
        return self._snapshot

    async def refresh(self) -> YoutubeCookieSnapshot:
        """Fetch, validate, and atomically materialize the current S3 cookie object."""
        async with self._lock:
            return await self._refresh_locked()

    async def refresh_after_rejection(self, snapshot: YoutubeCookieSnapshot) -> YoutubeCookieSnapshot:
        """Refresh only when the rejected operation used the current cookie generation."""
        if snapshot._store_identity is not self._store_identity:
            raise ValueError('YouTube cookie snapshot belongs to another store')
        async with self._lock:
            if snapshot._generation != self._generation:
                try:
                    return self.current()
                except YoutubeCookieNotMaterializedError:
                    return await self._refresh_locked()
            return await self._refresh_locked()

    async def upload(self, source: Path) -> int:
        """Validate and upload one selected local Netscape cookie file.

        The active runtime file is intentionally left unchanged.
        """
        try:
            data = source.read_bytes()
        except OSError as error:
            raise YoutubeCookieStoreError('failed to read YouTube cookie upload source') from error

        _validate_netscape_cookie_file(data)
        try:
            await self._s3_client.put_bytes(
                _COOKIE_KEY,
                data,
                content_type=S3ContentType.PLAIN,
            )
        except Exception as error:
            raise YoutubeCookieStoreError('failed to upload YouTube cookie file') from error
        return len(data)

    async def _refresh_locked(self) -> YoutubeCookieSnapshot:
        runtime_directory = _validate_runtime_directory_for_materialization(self._runtime_directory)
        try:
            data = await self._s3_client.get_bytes(_COOKIE_KEY)
        except Exception as error:
            raise YoutubeCookieStoreError('failed to retrieve YouTube cookie file') from error

        _validate_netscape_cookie_file(data)
        local_path = runtime_directory / _LOCAL_FILENAME
        _materialize_cookie_file(runtime_directory=runtime_directory, path=local_path, data=data)
        self._generation += 1
        self._snapshot = YoutubeCookieSnapshot(
            path=local_path,
            _generation=self._generation,
            _store_identity=self._store_identity,
        )
        return self._snapshot


def _validate_netscape_cookie_file(data: bytes) -> None:
    if not data:
        raise YoutubeCookieValidationError('YouTube cookie file is empty')
    try:
        text = data.decode('utf-8')
    except UnicodeDecodeError as error:
        raise YoutubeCookieValidationError('YouTube cookie file is not valid UTF-8') from error

    lines = text.splitlines()
    if not lines or lines[0].removeprefix('\ufeff') != _NETSCAPE_HEADER:
        raise YoutubeCookieValidationError('YouTube cookie file has an invalid Netscape header')

    record_count = 0
    for line in lines[1:]:
        if not line:
            continue
        if line.startswith('#HttpOnly_'):
            record = line.removeprefix('#HttpOnly_')
        elif line.startswith('#'):
            continue
        else:
            record = line

        fields = record.split('\t')
        if len(fields) != 7:
            raise YoutubeCookieValidationError('YouTube cookie file contains a malformed Netscape record')
        domain, include_subdomains, path, secure, expiry, name, _value = fields
        if (
            not domain
            or include_subdomains not in {'TRUE', 'FALSE'}
            or not path.startswith('/')
            or secure not in {'TRUE', 'FALSE'}
            or not expiry.isdigit()
            or not name
        ):
            raise YoutubeCookieValidationError('YouTube cookie file contains an invalid Netscape record')
        record_count += 1

    if record_count == 0:
        raise YoutubeCookieValidationError('YouTube cookie file contains no cookie records')


def _normalize_runtime_directory(runtime_directory: Path) -> Path:
    if not runtime_directory.is_absolute():
        raise YoutubeCookieMaterializationError('YouTube cookie runtime directory must be absolute')
    runtime_directory = Path(os.path.abspath(runtime_directory))
    if runtime_directory in _UNSAFE_RUNTIME_DIRECTORIES:
        raise YoutubeCookieMaterializationError('unsafe YouTube cookie runtime directory')
    return runtime_directory


def _validate_runtime_directory_for_materialization(runtime_directory: Path) -> Path:
    try:
        runtime_directory = runtime_directory.resolve(strict=False)
        exists = runtime_directory.exists()
    except OSError as error:
        raise YoutubeCookieMaterializationError('failed to inspect YouTube cookie runtime directory') from error
    if runtime_directory in _UNSAFE_RUNTIME_DIRECTORIES:
        raise YoutubeCookieMaterializationError('unsafe YouTube cookie runtime directory')
    if not exists:
        return runtime_directory
    try:
        if not runtime_directory.is_dir():
            raise YoutubeCookieMaterializationError('YouTube cookie runtime path is not a directory')
        if runtime_directory.stat().st_uid != os.geteuid():
            raise YoutubeCookieMaterializationError('YouTube cookie runtime directory is not owned by this process')
        if not os.access(runtime_directory, os.W_OK | os.X_OK):
            raise YoutubeCookieMaterializationError('YouTube cookie runtime directory is not writable')
    except OSError as error:
        raise YoutubeCookieMaterializationError('failed to inspect YouTube cookie runtime directory') from error
    return runtime_directory


def _materialize_cookie_file(*, runtime_directory: Path, path: Path, data: bytes) -> None:
    temporary_path: Path | None = None
    try:
        runtime_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        if not runtime_directory.is_dir():
            raise OSError('runtime path is not a directory')
        os.chmod(runtime_directory, 0o700)
        with tempfile.NamedTemporaryFile(
            mode='wb',
            dir=runtime_directory,
            prefix=f'.{path.name}.',
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(data)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.chmod(temporary_path, 0o600)
        os.replace(temporary_path, path)
        temporary_path = None
    except OSError as error:
        raise YoutubeCookieMaterializationError('failed to materialize YouTube cookie file') from error
    finally:
        if temporary_path is not None:
            with contextlib.suppress(OSError):
                temporary_path.unlink()
