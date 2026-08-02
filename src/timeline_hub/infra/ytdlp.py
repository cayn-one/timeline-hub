import asyncio
import contextlib
import json
import os
import shutil
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Coroutine, Iterator, Sequence
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import TypeVar

from loguru import logger


@dataclass(frozen=True, slots=True)
class TrackMetadata:
    artists: tuple[str, ...]
    title: str


class YtDlpMetadataError(RuntimeError):
    """Raised when yt-dlp metadata extraction or parsing fails."""


class YtDlpAuthenticationError(RuntimeError):
    """Raised when yt-dlp reports a known YouTube authentication rejection."""


class YtDlpCookieFileError(RuntimeError):
    """Raised when an operation-local yt-dlp cookie file cannot be prepared."""


@dataclass(frozen=True, slots=True)
class DownloadedAudio:
    audio: bytes
    cover: bytes | None = None
    metadata: TrackMetadata | None = None


@dataclass(frozen=True, slots=True)
class UrlTrackInfo:
    cover: bytes | None = None
    metadata: TrackMetadata | None = None


_YT_DLP_SENSITIVE_FLAGS = {
    '-p',
    '-u',
    '--add-headers',
    '--cookies',
    '--cookies-from-browser',
    '--extractor-args',
    '--netrc',
    '--netrc-cmd',
    '--netrc-file',
    '--netrc-location',
    '--password',
    '--proxy',
    '--user-agent',
    '--username',
    '--video-password',
    '--xff',
}
_YT_DLP_MAX_DIAGNOSTIC_TEXT = 2048
_PROCESS_TERMINATION_GRACE_SECONDS = 1.0
_T = TypeVar('_T')
_YOUTUBE_AUTHENTICATION_FAILURE_PHRASES = (
    "sign in to confirm you're not a bot",
    'authentication cookies are no longer valid',
    'authentication cookies are invalid',
    'authentication cookies have expired',
    'youtube account cookies are no longer valid',
    'youtube account cookies are invalid',
    'youtube account cookies have expired',
    'please sign in to youtube',
    'you must be logged in to youtube',
)


def _with_common_yt_dlp_args(
    args: Sequence[str],
    *,
    cookie_file: Path,
) -> tuple[str, ...]:
    _validate_cookie_file_path(cookie_file)
    command = tuple(args)
    common_args = (
        '--cookies',
        str(cookie_file),
        '--remote-components',
        'ejs:github',
        '--ignore-config',
    )
    return (command[0], *common_args, *command[1:])


def _validate_cookie_file_path(cookie_file: Path) -> None:
    if not cookie_file.is_absolute():
        raise ValueError('cookie_file must be an absolute path')


@contextlib.contextmanager
def _isolated_cookie_file(cookie_file: Path) -> Iterator[Path]:
    """Isolate one operation because yt-dlp writes its cookie jar on shutdown.

    Deletion is best effort: a cleanup failure is logged without masking the
    operation's result or cancellation.
    """
    _validate_cookie_file_path(cookie_file)
    temporary_path: Path | None = None
    try:
        try:
            with cookie_file.open('rb') as source_file:
                with tempfile.NamedTemporaryFile(
                    mode='wb',
                    dir=cookie_file.parent,
                    prefix=f'.{cookie_file.name}.operation.',
                    delete=False,
                ) as temporary_file:
                    temporary_path = Path(temporary_file.name)
                    shutil.copyfileobj(source_file, temporary_file)
            os.chmod(temporary_path, 0o600)
        except OSError as error:
            raise YtDlpCookieFileError('failed to prepare isolated yt-dlp cookie file') from error
        yield temporary_path
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except OSError:
                logger.warning('failed to remove isolated yt-dlp cookie file')


def _sanitize_yt_dlp_args(args: Sequence[str]) -> list[str]:
    sanitized: list[str] = []
    skip_next = False
    for arg in args:
        if skip_next:
            sanitized.append('<redacted>')
            skip_next = False
            continue
        flag, has_value, _value = _split_command_option(arg)
        if flag in _YT_DLP_SENSITIVE_FLAGS:
            if has_value:
                sanitized.append(f'{flag}=<redacted>')
            else:
                sanitized.append(flag)
                skip_next = True
            continue
        sanitized.append(_sanitize_command_token(arg))
    return sanitized


def _split_command_option(arg: str) -> tuple[str, bool, str]:
    if '=' not in arg or not arg.startswith('-'):
        return arg, False, ''
    flag, value = arg.split('=', 1)
    return flag, True, value


def _sanitize_command_token(token: str) -> str:
    if not token.startswith(('http://', 'https://')):
        return token
    parsed = urllib.parse.urlsplit(token)
    if parsed.scheme not in {'http', 'https'} or not parsed.netloc:
        return token
    netloc = parsed.hostname or ''
    if parsed.port is not None:
        netloc = f'{netloc}:{parsed.port}'
    return urllib.parse.urlunsplit((parsed.scheme, netloc, parsed.path, '', ''))


def _truncate_diagnostic_text(text: str, *, max_length: int = _YT_DLP_MAX_DIAGNOSTIC_TEXT) -> str:
    if len(text) <= max_length:
        return text
    omitted = len(text) - max_length
    marker = f'... <truncated {omitted} chars> ...'
    if max_length <= len(marker):
        return marker[:max_length]
    remaining = max_length - len(marker)
    prefix_length = remaining // 2
    suffix_length = remaining - prefix_length
    return f'{text[:prefix_length]}{marker}{text[-suffix_length:]}'


def _is_youtube_authentication_failure(stderr: bytes) -> bool:
    text = stderr.decode(errors='replace').lower().replace('\u2019', "'")
    if any(phrase in text for phrase in _YOUTUBE_AUTHENTICATION_FAILURE_PHRASES):
        return True
    return 'login required' in text and any(
        youtube_context in text
        for youtube_context in (
            '[youtube',
            'youtube.com',
            'youtube account',
        )
    )


def _format_yt_dlp_failure(
    *,
    operation: str,
    returncode: int,
    args: Sequence[str],
    stderr: bytes,
    stdout: bytes = b'',
) -> str:
    stderr_text = _truncate_diagnostic_text(stderr.decode(errors='replace').strip())
    stdout_text = _truncate_diagnostic_text(stdout.decode(errors='replace').strip())
    message = (
        f'yt-dlp {operation} failed with exit code {returncode}; '
        f'command={_sanitize_yt_dlp_args(args)}; '
        f'stderr={stderr_text}'
    )
    if stdout_text:
        message += f'; stdout={stdout_text}'
    return message


def _format_process_failure(
    *,
    command_name: str,
    operation: str,
    returncode: int,
    args: Sequence[str],
    stderr: bytes,
    stdout: bytes = b'',
) -> str:
    stderr_text = _truncate_diagnostic_text(stderr.decode(errors='replace').strip())
    stdout_text = _truncate_diagnostic_text(stdout.decode(errors='replace').strip())
    message = (
        f'{command_name} {operation} failed with exit code {returncode}; '
        f'command={_sanitize_yt_dlp_args(args)}; '
        f'stderr={stderr_text}'
    )
    if stdout_text:
        message += f'; stdout={stdout_text}'
    return message


def _yt_dlp_timeout_error(*, operation: str, args: Sequence[str], timeout: timedelta) -> asyncio.TimeoutError:
    return asyncio.TimeoutError(
        f'yt-dlp {operation} timed out after {timeout.total_seconds()}s; command={_sanitize_yt_dlp_args(args)}'
    )


async def _stop_process(
    process: asyncio.subprocess.Process,
    *,
    grace_seconds: float | None = None,
) -> None:
    """Terminate, kill if needed, and reap one owned subprocess."""
    if grace_seconds is None:
        grace_seconds = _PROCESS_TERMINATION_GRACE_SECONDS
    if process.returncode is not None:
        await process.wait()
        return

    terminate = getattr(process, 'terminate', None)
    if callable(terminate):
        try:
            terminate()
        except ProcessLookupError:
            pass
    else:
        try:
            process.kill()
        except ProcessLookupError:
            pass

    wait_task = asyncio.create_task(process.wait())
    try:
        await asyncio.wait_for(asyncio.shield(wait_task), timeout=grace_seconds)
        return
    except asyncio.TimeoutError:
        pass

    if process.returncode is None:
        try:
            process.kill()
        except ProcessLookupError:
            pass
    await wait_task


async def _await_task_after_cancellation(task: asyncio.Task[_T]) -> tuple[_T, bool]:
    """Wait for a task despite repeated cancellation and report later cancellation."""
    cancelled_during_cleanup = False
    while True:
        try:
            return await asyncio.shield(task), cancelled_during_cleanup
        except asyncio.CancelledError:
            cancelled_during_cleanup = True
            if task.done():
                return task.result(), cancelled_during_cleanup


async def _await_cleanup_after_cancellation(cleanup: Coroutine[object, object, None]) -> bool:
    """Finish cleanup despite repeated cancellation and report a later cancellation."""
    _result, cancelled_during_cleanup = await _await_task_after_cancellation(asyncio.create_task(cleanup))
    return cancelled_during_cleanup


async def _create_owned_subprocess_exec(
    *args: str,
    stdin: int | None = None,
    stdout: int | None = None,
    stderr: int | None = None,
) -> asyncio.subprocess.Process:
    """Create a subprocess without losing ownership if the caller is cancelled."""
    creation_task = asyncio.create_task(
        asyncio.create_subprocess_exec(
            *args,
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
        ),
    )
    try:
        return await asyncio.shield(creation_task)
    except asyncio.CancelledError:
        try:
            process, _cancelled_during_creation = await _await_task_after_cancellation(creation_task)
        except asyncio.CancelledError:
            # The caller-visible cancellation remains the result when creation also stops.
            pass
        except Exception:
            # No child was created, so there is no cleanup failure to report.
            pass
        else:
            try:
                await _await_cleanup_after_cancellation(_stop_process(process))
            except asyncio.CancelledError:
                # Preserve the original caller cancellation after mandatory cleanup.
                pass
            except Exception:
                logger.warning('yt-dlp subprocess cleanup failed after cancellation during creation')
        raise


async def _run_yt_dlp_command(
    *,
    operation: str,
    args: Sequence[str],
    timeout: timedelta,
) -> tuple[bytes, bytes]:
    proc = await _create_owned_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=timeout.total_seconds(),
        )
    except asyncio.TimeoutError:
        try:
            cancelled_during_cleanup = await _await_cleanup_after_cancellation(_stop_process(proc))
            if cancelled_during_cleanup:
                raise asyncio.CancelledError
        except Exception as error:
            logger.warning('yt-dlp {} cleanup failed after timeout: {}', operation, error)
        raise _yt_dlp_timeout_error(operation=operation, args=args, timeout=timeout) from None
    except asyncio.CancelledError:
        try:
            await _await_cleanup_after_cancellation(_stop_process(proc))
        except Exception as error:
            logger.warning('yt-dlp {} cleanup failed after cancellation: {}', operation, error)
        raise

    returncode = proc.returncode if proc.returncode is not None else -1
    if returncode != 0:
        message = _format_yt_dlp_failure(
            operation=operation,
            returncode=returncode,
            args=args,
            stderr=stderr,
            stdout=stdout,
        )
        if _is_youtube_authentication_failure(stderr):
            raise YtDlpAuthenticationError(message)
        raise RuntimeError(message)

    return stdout, stderr


async def fetch_track_info(
    url: str,
    *,
    cookie_file: Path,
    with_cover: bool = True,
    with_metadata: bool = True,
    timeout: timedelta = timedelta(minutes=1),
) -> UrlTrackInfo:
    """Fetch URL-derived cover and metadata without downloading audio.

    Args:
        url: Source URL to inspect.
        with_cover: Whether to fetch cover bytes.
        with_metadata: Whether to fetch parsed track metadata.
        timeout: Maximum time allowed per `yt-dlp` subprocess run.

    Returns:
        UrlTrackInfo with optional cover and metadata.

    Raises:
        ValueError: If `url` is invalid.
        RuntimeError: If cover extraction fails or `yt-dlp` fails.
        YtDlpMetadataError: If metadata extraction or parsing fails.
    """
    _validate_cookie_file_path(cookie_file)
    normalized_url = _normalize_url(url)
    if not with_cover and not with_metadata:
        return UrlTrackInfo()

    with _isolated_cookie_file(cookie_file) as operation_cookie_file:
        cover: bytes | None = None
        metadata: TrackMetadata | None = None
        if with_cover:
            youtube_video_id = None
            if not with_metadata:
                youtube_video_id = _extract_youtube_video_id(normalized_url)
            if youtube_video_id is not None:
                cover = await _download_youtube_thumbnail_as_jpg(youtube_video_id, timeout=timeout)
            else:
                cover = await _download_cover_as_jpg(
                    normalized_url,
                    cookie_file=operation_cookie_file,
                    timeout=timeout,
                )
        if with_metadata:
            metadata = await _download_track_metadata(
                normalized_url,
                cookie_file=operation_cookie_file,
                timeout=timeout,
            )
        return UrlTrackInfo(cover=cover, metadata=metadata)


async def download_audio_as_opus(
    url: str,
    *,
    cookie_file: Path,
    with_cover: bool = False,
    with_metadata: bool = False,
    max_duration: timedelta | None = None,
    timeout: timedelta = timedelta(minutes=3),
) -> DownloadedAudio:
    """Download one URL audio track as Opus bytes using `yt-dlp`.

    Args:
        url: Source URL to download.
        max_duration: Optional maximum audio duration to return.
        timeout: Maximum time allowed for the `yt-dlp` subprocess run.

    Returns:
        DownloadedAudio with Opus audio, optional JPG cover, and optional parsed track metadata.

    Raises:
        ValueError: If `url` is invalid.
        RuntimeError: If `yt-dlp` fails or output validation fails.
        YtDlpMetadataError: If metadata is requested but cannot be extracted or parsed.
    """
    _validate_cookie_file_path(cookie_file)
    with _isolated_cookie_file(cookie_file) as operation_cookie_file:
        result = await _download_audio(
            url,
            cookie_file=operation_cookie_file,
            with_cover=with_cover,
            with_metadata=with_metadata,
            max_duration=max_duration,
            timeout=timeout,
        )
    if with_cover and result.cover is None:
        raise RuntimeError('yt-dlp did not produce cover output')
    return result


async def _download_audio(
    url: str,
    *,
    cookie_file: Path,
    with_cover: bool,
    with_metadata: bool,
    max_duration: timedelta | None,
    timeout: timedelta,
) -> DownloadedAudio:
    if max_duration is None:
        audio, cover, metadata = await _download_audio_as_opus_internal(
            url,
            cookie_file=cookie_file,
            download_cover=with_cover,
            with_metadata=with_metadata,
            timeout=timeout,
        )
        return DownloadedAudio(audio=audio, cover=cover, metadata=metadata)

    _validate_max_duration(max_duration)
    duration = await _get_media_duration(url, cookie_file=cookie_file, timeout=timedelta(seconds=30))
    if duration is not None and duration <= max_duration:
        audio, cover, metadata = await _download_audio_as_opus_internal(
            url,
            cookie_file=cookie_file,
            download_cover=with_cover,
            with_metadata=with_metadata,
            timeout=timeout,
        )
        return DownloadedAudio(audio=audio, cover=cover, metadata=metadata)

    audio = await _download_audio_as_opus_clipped(
        url,
        cookie_file=cookie_file,
        max_duration=max_duration,
        timeout=timeout,
    )
    metadata: TrackMetadata | None = None
    if with_metadata:
        metadata = await _download_track_metadata(url, cookie_file=cookie_file, timeout=timeout)
    if with_cover:
        cover = await _download_cover_as_jpg(url, cookie_file=cookie_file, timeout=timeout)
        return DownloadedAudio(audio=audio, cover=cover, metadata=metadata)
    return DownloadedAudio(audio=audio, metadata=metadata)


async def get_media_duration(
    url: str,
    *,
    cookie_file: Path,
    timeout: timedelta = timedelta(seconds=30),
) -> timedelta | None:
    with _isolated_cookie_file(cookie_file) as operation_cookie_file:
        return await _get_media_duration(url, cookie_file=operation_cookie_file, timeout=timeout)


async def _get_media_duration(
    url: str,
    *,
    cookie_file: Path,
    timeout: timedelta,
) -> timedelta | None:
    normalized_url = _normalize_url(url)
    stdout, _stderr = await _run_yt_dlp_command(
        operation='duration_probe',
        args=_with_common_yt_dlp_args(
            (
                'yt-dlp',
                '--print',
                '%(duration)s',
                '--simulate',
                '--skip-download',
                '--ignore-no-formats-error',
                '--no-playlist',
                normalized_url,
            ),
            cookie_file=cookie_file,
        ),
        timeout=timeout,
    )

    duration_text = stdout.decode(errors='replace').strip()
    if not duration_text or duration_text in {'NA', 'None'}:
        return None
    try:
        duration_seconds = float(duration_text)
    except ValueError:
        return None
    if duration_seconds <= 0:
        return None
    return timedelta(seconds=duration_seconds)


def _normalize_url(url: str) -> str:
    if not isinstance(url, str):
        raise ValueError('url must be a string')

    normalized_url = url.strip()
    if not normalized_url:
        raise ValueError('url must not be empty')
    return normalized_url


def _validate_max_duration(max_duration: timedelta) -> None:
    if not isinstance(max_duration, timedelta):
        raise ValueError('max_duration must be a timedelta')
    if max_duration <= timedelta(0):
        raise ValueError('max_duration must be > 0')


def _extract_youtube_video_id(url: str) -> str | None:
    parsed = urllib.parse.urlparse(url)
    host = parsed.netloc.lower()
    if host in ('youtube.com', 'www.youtube.com', 'music.youtube.com'):
        if parsed.path == '/watch':
            video_ids = urllib.parse.parse_qs(parsed.query).get('v', ())
            for video_id in video_ids:
                normalized_video_id = video_id.strip()
                if _is_valid_youtube_video_id(normalized_video_id):
                    return normalized_video_id
            return None
        path_segments = [segment for segment in parsed.path.split('/') if segment]
        if len(path_segments) >= 2 and path_segments[0] == 'shorts':
            short_video_id = path_segments[1].strip()
            if _is_valid_youtube_video_id(short_video_id):
                return short_video_id
            return None
        return None

    if host == 'youtu.be':
        path_segments = [segment for segment in parsed.path.split('/') if segment]
        if len(path_segments) != 1:
            return None
        short_video_id = path_segments[0].strip()
        if _is_valid_youtube_video_id(short_video_id):
            return short_video_id
    return None


def _is_valid_youtube_video_id(value: str) -> bool:
    if len(value) != 11:
        return False
    return all(character.isalnum() or character in {'-', '_'} for character in value)


async def _download_youtube_thumbnail_as_jpg(
    video_id: str,
    *,
    timeout: timedelta,
) -> bytes:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout.total_seconds()

    thumbnail_names = ('maxresdefault.jpg', 'sddefault.jpg', 'hqdefault.jpg')
    for thumbnail_name in thumbnail_names:
        remaining_seconds = deadline - loop.time()
        if remaining_seconds <= 0:
            raise asyncio.TimeoutError
        thumbnail_url = f'https://i.ytimg.com/vi/{video_id}/{thumbnail_name}'
        try:
            cover_bytes = await _download_http_bytes(
                thumbnail_url,
                timeout_seconds=remaining_seconds,
            )
        except urllib.error.URLError, TimeoutError:
            continue
        if not cover_bytes:
            continue
        if not cover_bytes.startswith(b'\xff\xd8'):
            continue
        return cover_bytes

    raise RuntimeError('youtube thumbnail did not produce cover output')


async def _download_http_bytes(url: str, *, timeout_seconds: float) -> bytes:
    if timeout_seconds <= 0:
        raise asyncio.TimeoutError

    def _fetch() -> bytes:
        with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
            status = response.getcode()
            if status != 200:
                raise RuntimeError(f'http download failed with status {status}')
            return response.read()

    return await asyncio.to_thread(_fetch)


async def _download_audio_as_opus_internal(
    url: str,
    *,
    cookie_file: Path,
    download_cover: bool,
    with_metadata: bool,
    timeout: timedelta,
) -> tuple[bytes, bytes | None, TrackMetadata | None]:
    normalized_url = _normalize_url(url)

    with tempfile.TemporaryDirectory() as temp_dir:
        output_template = Path(temp_dir) / 'audio.%(ext)s'
        args: list[str] = [
            'yt-dlp',
            '-f',
            'bestaudio',
            '--extract-audio',
            '--audio-format',
            'opus',
            '--no-playlist',
        ]
        if download_cover:
            args.extend(
                [
                    '--write-thumbnail',
                    '--convert-thumbnails',
                    'jpg',
                ]
            )
        if with_metadata:
            args.append('--write-info-json')
        args.extend(['-o', str(output_template), normalized_url])

        await _run_yt_dlp_command(
            operation='audio_download',
            args=_with_common_yt_dlp_args(args, cookie_file=cookie_file),
            timeout=timeout,
        )
        output_files = sorted(Path(temp_dir).glob('*.opus'))
        if not output_files:
            raise RuntimeError('yt-dlp did not produce opus output')
        if len(output_files) > 1:
            raise RuntimeError('yt-dlp produced multiple opus outputs')

        audio_bytes = output_files[0].read_bytes()
        if not audio_bytes.startswith(b'OggS'):
            raise RuntimeError('yt-dlp output is not a valid Ogg/Opus container')

        metadata: TrackMetadata | None = None
        if with_metadata:
            metadata_files = sorted(Path(temp_dir).glob('*.info.json'))
            if not metadata_files:
                raise YtDlpMetadataError('yt-dlp did not produce metadata output')
            if len(metadata_files) > 1:
                raise YtDlpMetadataError('yt-dlp produced multiple metadata outputs')
            try:
                metadata_obj = json.loads(metadata_files[0].read_text())
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
                raise YtDlpMetadataError('yt-dlp produced invalid metadata output') from error
            if not isinstance(metadata_obj, dict):
                raise YtDlpMetadataError('yt-dlp produced invalid metadata output')
            metadata = _parse_track_metadata(metadata_obj)

        if not download_cover:
            return audio_bytes, None, metadata

        cover_files = sorted(Path(temp_dir).glob('*.jpg'))
        if not cover_files:
            return audio_bytes, None, metadata
        if len(cover_files) > 1:
            raise RuntimeError('yt-dlp produced multiple cover outputs')

        cover_bytes = cover_files[0].read_bytes()
        if not cover_bytes:
            raise RuntimeError('yt-dlp produced empty cover output')
        return audio_bytes, cover_bytes, metadata


def _parse_track_metadata(raw_metadata: dict[str, object]) -> TrackMetadata:
    title_candidates = (raw_metadata.get('track'), raw_metadata.get('title'))
    title = ''
    for candidate in title_candidates:
        if isinstance(candidate, str):
            stripped = candidate.strip()
            if stripped:
                title = stripped
                break

    artists: tuple[str, ...] = ()
    artists_value = raw_metadata.get('artists')
    if isinstance(artists_value, list):
        parsed_artists: list[str] = []
        for value in artists_value:
            if isinstance(value, str):
                stripped = value.strip()
                if stripped:
                    parsed_artists.append(stripped)
        if parsed_artists:
            artists = tuple(parsed_artists)

    if not artists:
        artists = _parse_comma_separated_artists(raw_metadata.get('artist'))
    if not artists:
        artists = _parse_comma_separated_artists(raw_metadata.get('creator'))

    if not title or not artists:
        raise YtDlpMetadataError('yt-dlp produced incomplete metadata output')

    return TrackMetadata(artists=artists, title=title)


def _parse_comma_separated_artists(value: object) -> tuple[str, ...]:
    if not isinstance(value, str):
        return ()
    artists = tuple(part.strip() for part in value.split(',') if part.strip())
    return artists


async def _download_cover_as_jpg(
    url: str,
    *,
    cookie_file: Path,
    timeout: timedelta,
) -> bytes:
    normalized_url = _normalize_url(url)
    with tempfile.TemporaryDirectory() as temp_dir:
        output_template = Path(temp_dir) / 'cover.%(ext)s'
        await _run_yt_dlp_command(
            operation='cover_download',
            args=_with_common_yt_dlp_args(
                (
                    'yt-dlp',
                    '--skip-download',
                    '--write-thumbnail',
                    '--convert-thumbnails',
                    'jpg',
                    '-o',
                    str(output_template),
                    normalized_url,
                ),
                cookie_file=cookie_file,
            ),
            timeout=timeout,
        )

        cover_files = sorted(Path(temp_dir).glob('*.jpg'))
        if not cover_files:
            raise RuntimeError('yt-dlp did not produce cover output')
        if len(cover_files) > 1:
            raise RuntimeError('yt-dlp produced multiple cover outputs')

        cover_bytes = cover_files[0].read_bytes()
        if not cover_bytes:
            raise RuntimeError('yt-dlp produced empty cover output')
        return cover_bytes


async def _download_track_metadata(
    url: str,
    *,
    cookie_file: Path,
    timeout: timedelta,
) -> TrackMetadata:
    normalized_url = _normalize_url(url)
    with tempfile.TemporaryDirectory() as temp_dir:
        output_template = Path(temp_dir) / 'metadata.%(ext)s'
        await _run_yt_dlp_command(
            operation='metadata_download',
            args=_with_common_yt_dlp_args(
                (
                    'yt-dlp',
                    '--skip-download',
                    '--write-info-json',
                    '-o',
                    str(output_template),
                    normalized_url,
                ),
                cookie_file=cookie_file,
            ),
            timeout=timeout,
        )

        metadata_files = sorted(Path(temp_dir).glob('*.info.json'))
        if not metadata_files:
            raise YtDlpMetadataError('yt-dlp did not produce metadata output')
        if len(metadata_files) > 1:
            raise YtDlpMetadataError('yt-dlp produced multiple metadata outputs')
        try:
            metadata_obj = json.loads(metadata_files[0].read_text())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise YtDlpMetadataError('yt-dlp produced invalid metadata output') from error
        if not isinstance(metadata_obj, dict):
            raise YtDlpMetadataError('yt-dlp produced invalid metadata output')
        return _parse_track_metadata(metadata_obj)


async def _download_audio_as_opus_clipped(
    url: str,
    *,
    cookie_file: Path,
    max_duration: timedelta,
    timeout: timedelta,
) -> bytes:
    normalized_url = _normalize_url(url)
    _validate_max_duration(max_duration)
    max_duration_seconds = str(max_duration.total_seconds())
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout.total_seconds()
    source_codec = await _get_selected_audio_codec(
        normalized_url,
        cookie_file=cookie_file,
        timeout=timeout,
    )
    remaining_timeout_seconds = deadline - loop.time()
    if remaining_timeout_seconds <= 0:
        raise asyncio.TimeoutError

    codec_args: tuple[str, ...]
    if not _codec_is_opus(source_codec):
        codec_args = (
            '-c:a',
            'libopus',
            '-b:a',
            '160k',
            '-vbr',
            'on',
            '-compression_level',
            '10',
        )
    else:
        codec_args = (
            '-c:a',
            'copy',
        )

    ytdlp_args = _with_common_yt_dlp_args(
        (
            'yt-dlp',
            '-f',
            'bestaudio',
            '--no-playlist',
            '-o',
            '-',
            normalized_url,
        ),
        cookie_file=cookie_file,
    )
    ytdlp_proc = await _create_owned_subprocess_exec(
        *ytdlp_args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    ffmpeg_args = (
        'ffmpeg',
        '-hide_banner',
        '-loglevel',
        'error',
        '-nostats',
        '-nostdin',
        '-y',
        '-threads',
        '1',
        '-i',
        'pipe:0',
        '-t',
        max_duration_seconds,
        '-vn',
        *codec_args,
        '-f',
        'opus',
        'pipe:1',
    )
    try:
        ffmpeg_proc = await _create_owned_subprocess_exec(
            *ffmpeg_args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except asyncio.CancelledError:
        try:
            await _await_cleanup_after_cancellation(_stop_process(ytdlp_proc))
        except Exception as error:
            logger.warning('yt-dlp clipped_audio_download cleanup failed after cancellation: {}', error)
        raise
    except Exception:
        await _stop_process(ytdlp_proc)
        raise
    assert ytdlp_proc.stdout is not None
    assert ytdlp_proc.stderr is not None
    assert ffmpeg_proc.stdin is not None
    assert ffmpeg_proc.stdout is not None
    assert ffmpeg_proc.stderr is not None
    pipe_task = asyncio.create_task(_pipe_stream(ytdlp_proc.stdout, ffmpeg_proc.stdin))
    ffmpeg_stdout_task = asyncio.create_task(ffmpeg_proc.stdout.read())
    ffmpeg_stderr_task = asyncio.create_task(ffmpeg_proc.stderr.read())
    ytdlp_stderr_task = asyncio.create_task(ytdlp_proc.stderr.read())
    ffmpeg_wait_task = asyncio.create_task(ffmpeg_proc.wait())
    ytdlp_wait_task = asyncio.create_task(ytdlp_proc.wait())
    tasks = (
        pipe_task,
        ffmpeg_stdout_task,
        ffmpeg_stderr_task,
        ytdlp_stderr_task,
        ffmpeg_wait_task,
        ytdlp_wait_task,
    )
    ytdlp_returncode: int | None = None
    ytdlp_stderr = b''
    pipeline_settled = False

    async def _stop_pipeline() -> None:
        cleanup_error: Exception | None = None
        for process in (ffmpeg_proc, ytdlp_proc):
            try:
                await _stop_process(process)
            except Exception as error:
                if cleanup_error is None:
                    cleanup_error = error
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if cleanup_error is not None:
            raise cleanup_error

    try:
        async with asyncio.timeout(remaining_timeout_seconds):
            await ffmpeg_wait_task
            ffmpeg_returncode = ffmpeg_wait_task.result()
            ffmpeg_stdout = await ffmpeg_stdout_task
            ffmpeg_stderr = await ffmpeg_stderr_task

            if ytdlp_wait_task.done():
                ytdlp_returncode = ytdlp_wait_task.result()
            else:
                try:
                    ytdlp_returncode = await asyncio.wait_for(
                        asyncio.shield(ytdlp_wait_task),
                        timeout=1.0,
                    )
                except asyncio.TimeoutError:
                    await _stop_process(ytdlp_proc)
                    ytdlp_returncode = await ytdlp_wait_task
                except asyncio.CancelledError:
                    raise
            if ytdlp_stderr_task.done():
                ytdlp_stderr = ytdlp_stderr_task.result()
            else:
                try:
                    ytdlp_stderr = await asyncio.wait_for(
                        asyncio.shield(ytdlp_stderr_task),
                        timeout=1.0,
                    )
                except asyncio.TimeoutError:
                    ytdlp_stderr = b''
    except asyncio.TimeoutError:
        try:
            cancelled_during_cleanup = await _await_cleanup_after_cancellation(_stop_pipeline())
            pipeline_settled = True
            if cancelled_during_cleanup:
                raise asyncio.CancelledError
        except Exception as error:
            logger.warning('yt-dlp clipped_audio_download cleanup failed after timeout: {}', error)
        raise _yt_dlp_timeout_error(operation='clipped_audio_download', args=ytdlp_args, timeout=timeout) from None
    except asyncio.CancelledError:
        try:
            await _await_cleanup_after_cancellation(_stop_pipeline())
            pipeline_settled = True
        except Exception as error:
            logger.warning('yt-dlp clipped_audio_download cleanup failed after cancellation: {}', error)
        raise
    except Exception:
        await _stop_pipeline()
        pipeline_settled = True
        raise
    finally:
        if not pipeline_settled:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    ytdlp_failure_message: str | None = None
    if ytdlp_returncode not in (None, 0):
        ytdlp_stderr_text = ytdlp_stderr.decode(errors='replace')
        if 'broken pipe' not in ytdlp_stderr_text.lower():
            ytdlp_failure_message = _format_yt_dlp_failure(
                operation='clipped_audio_download',
                returncode=ytdlp_returncode,
                args=ytdlp_args,
                stderr=ytdlp_stderr,
            )

    ffmpeg_failure_message: str | None = None
    if ffmpeg_returncode != 0:
        ffmpeg_failure_message = _format_process_failure(
            command_name='ffmpeg',
            operation='clipped_audio_download',
            returncode=ffmpeg_returncode,
            args=ffmpeg_args,
            stderr=ffmpeg_stderr,
            stdout=ffmpeg_stdout,
        )

    if ytdlp_failure_message is not None:
        if _is_youtube_authentication_failure(ytdlp_stderr):
            raise YtDlpAuthenticationError(ytdlp_failure_message)
        raise RuntimeError(ytdlp_failure_message)
    if ffmpeg_failure_message is not None:
        raise RuntimeError(ffmpeg_failure_message)

    if not ffmpeg_stdout or not ffmpeg_stdout.startswith(b'OggS'):
        raise RuntimeError('yt-dlp output is not a valid Ogg/Opus container')
    return ffmpeg_stdout


async def _pipe_stream(
    source: asyncio.StreamReader,
    destination: asyncio.StreamWriter,
) -> None:
    try:
        while True:
            chunk = await source.read(65536)
            if not chunk:
                break
            destination.write(chunk)
            await destination.drain()
    except BrokenPipeError, ConnectionResetError:
        return
    finally:
        try:
            destination.close()
            await destination.wait_closed()
        except BrokenPipeError, ConnectionResetError:
            pass


async def _get_selected_audio_codec(
    url: str,
    *,
    cookie_file: Path,
    timeout: timedelta = timedelta(seconds=30),
) -> str | None:
    stdout, _stderr = await _run_yt_dlp_command(
        operation='codec_probe',
        args=_with_common_yt_dlp_args(
            (
                'yt-dlp',
                '--print',
                '%(acodec)s',
                '-f',
                'bestaudio',
                '--skip-download',
                '--no-playlist',
                url,
            ),
            cookie_file=cookie_file,
        ),
        timeout=timeout,
    )

    codec = stdout.decode(errors='replace').strip()
    if not codec or codec.lower() in {'none', 'na', 'unknown'}:
        return None
    return codec


def _codec_is_opus(codec: str | None) -> bool:
    if codec is None:
        return False
    normalized = codec.strip().lower()
    if not normalized:
        return False
    if normalized in {'none', 'na', 'unknown'}:
        return False
    return normalized == 'opus' or normalized.startswith('opus')
