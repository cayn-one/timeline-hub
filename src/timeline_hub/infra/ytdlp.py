import asyncio
import contextlib
import json
import tempfile
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TrackMetadata:
    artists: tuple[str, ...]
    title: str


class YtDlpMetadataError(RuntimeError):
    """Raised when yt-dlp metadata extraction or parsing fails."""


@dataclass(frozen=True, slots=True)
class DownloadedAudio:
    audio: bytes
    cover: bytes | None = None
    metadata: TrackMetadata | None = None


async def download_audio_as_opus(
    url: str,
    *,
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
    result = await _download_audio(
        url,
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
    with_cover: bool,
    with_metadata: bool,
    max_duration: timedelta | None,
    timeout: timedelta,
) -> DownloadedAudio:
    if max_duration is None:
        audio, cover, metadata = await _download_audio_as_opus_internal(
            url,
            download_cover=with_cover,
            with_metadata=with_metadata,
            timeout=timeout,
        )
        return DownloadedAudio(audio=audio, cover=cover, metadata=metadata)

    _validate_max_duration(max_duration)
    duration = await get_media_duration(url, timeout=timedelta(seconds=30))
    if duration is not None and duration <= max_duration:
        audio, cover, metadata = await _download_audio_as_opus_internal(
            url,
            download_cover=with_cover,
            with_metadata=with_metadata,
            timeout=timeout,
        )
        return DownloadedAudio(audio=audio, cover=cover, metadata=metadata)

    audio = await _download_audio_as_opus_clipped(
        url,
        max_duration=max_duration,
        timeout=timeout,
    )
    metadata: TrackMetadata | None = None
    if with_metadata:
        metadata = await _download_track_metadata(url, timeout=timeout)
    if with_cover:
        cover = await _download_cover_as_jpg(url, timeout=timeout)
        return DownloadedAudio(audio=audio, cover=cover, metadata=metadata)
    return DownloadedAudio(audio=audio, metadata=metadata)


async def get_media_duration(
    url: str,
    *,
    timeout: timedelta = timedelta(seconds=30),
) -> timedelta | None:
    normalized_url = _normalize_url(url)
    proc = await asyncio.create_subprocess_exec(
        'yt-dlp',
        '--no-warnings',
        '--print',
        '%(duration)s',
        '--skip-download',
        '--no-playlist',
        normalized_url,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=timeout.total_seconds(),
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise

    if proc.returncode != 0:
        stderr_text = stderr.decode(errors='replace')
        raise RuntimeError(f'yt-dlp failed: {stderr_text}')

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


async def _download_audio_as_opus_internal(
    url: str,
    *,
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
            '--quiet',
            '--no-warnings',
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
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            _, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout.total_seconds(),
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise

        if proc.returncode != 0:
            stderr_text = stderr.decode(errors='replace')
            raise RuntimeError(f'yt-dlp failed: {stderr_text}')

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
    timeout: timedelta,
) -> bytes:
    normalized_url = _normalize_url(url)
    with tempfile.TemporaryDirectory() as temp_dir:
        output_template = Path(temp_dir) / 'cover.%(ext)s'
        proc = await asyncio.create_subprocess_exec(
            'yt-dlp',
            '--skip-download',
            '--write-thumbnail',
            '--convert-thumbnails',
            'jpg',
            '--quiet',
            '--no-warnings',
            '--no-playlist',
            '-o',
            str(output_template),
            normalized_url,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            _, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout.total_seconds(),
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise

        if proc.returncode != 0:
            stderr_text = stderr.decode(errors='replace')
            raise RuntimeError(f'yt-dlp failed: {stderr_text}')

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
    timeout: timedelta,
) -> TrackMetadata:
    normalized_url = _normalize_url(url)
    with tempfile.TemporaryDirectory() as temp_dir:
        output_template = Path(temp_dir) / 'metadata.%(ext)s'
        proc = await asyncio.create_subprocess_exec(
            'yt-dlp',
            '--skip-download',
            '--write-info-json',
            '--quiet',
            '--no-warnings',
            '--no-playlist',
            '-o',
            str(output_template),
            normalized_url,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            _, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout.total_seconds(),
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise

        if proc.returncode != 0:
            stderr_text = stderr.decode(errors='replace')
            raise RuntimeError(f'yt-dlp failed: {stderr_text}')

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
    max_duration: timedelta,
    timeout: timedelta,
) -> bytes:
    async def _stop_process(proc: asyncio.subprocess.Process, *, grace_seconds: float) -> None:
        if proc.returncode is not None:
            return
        terminate = getattr(proc, 'terminate', None)
        if callable(terminate):
            with contextlib.suppress(ProcessLookupError):
                terminate()
        else:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
        try:
            await asyncio.wait_for(proc.wait(), timeout=grace_seconds)
            return
        except asyncio.TimeoutError:
            pass
        if proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
        with contextlib.suppress(Exception):
            await proc.wait()

    normalized_url = _normalize_url(url)
    _validate_max_duration(max_duration)
    max_duration_seconds = str(max_duration.total_seconds())
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout.total_seconds()
    source_codec = await _get_selected_audio_codec(
        normalized_url,
        timeout=timeout,
    )
    remaining_timeout_seconds = deadline - loop.time()
    if remaining_timeout_seconds <= 0:
        raise asyncio.TimeoutError
    codec_is_opus = _codec_is_opus(source_codec)
    codec_args: tuple[str, ...]
    if codec_is_opus:
        codec_args = (
            '-c:a',
            'copy',
        )
    else:
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

    ytdlp_proc = await asyncio.create_subprocess_exec(
        'yt-dlp',
        '-f',
        'bestaudio',
        '--quiet',
        '--no-warnings',
        '--no-playlist',
        '-o',
        '-',
        normalized_url,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    ffmpeg_proc = await asyncio.create_subprocess_exec(
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
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
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
    try:
        async with asyncio.timeout(remaining_timeout_seconds):
            await ffmpeg_wait_task
            ffmpeg_returncode = ffmpeg_wait_task.result()
            ffmpeg_stdout = await ffmpeg_stdout_task
            ffmpeg_stderr = await ffmpeg_stderr_task

            if ffmpeg_returncode == 0 and ffmpeg_stdout and ffmpeg_stdout.startswith(b'OggS'):
                await _stop_process(ytdlp_proc, grace_seconds=1.0)
                return ffmpeg_stdout

            ytdlp_returncode = ytdlp_proc.returncode
            if ytdlp_returncode is None:
                try:
                    ytdlp_returncode = await asyncio.wait_for(ytdlp_wait_task, timeout=1.0)
                except asyncio.TimeoutError:
                    ytdlp_returncode = None
            if ytdlp_stderr_task.done():
                ytdlp_stderr = ytdlp_stderr_task.result()
            else:
                try:
                    ytdlp_stderr = await asyncio.wait_for(ytdlp_stderr_task, timeout=1.0)
                except asyncio.TimeoutError:
                    ytdlp_stderr = b''
    except asyncio.TimeoutError:
        await _stop_process(ytdlp_proc, grace_seconds=0.1)
        await _stop_process(ffmpeg_proc, grace_seconds=0.1)
        await asyncio.gather(
            ytdlp_proc.wait(),
            ffmpeg_proc.wait(),
            return_exceptions=True,
        )
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    if ytdlp_returncode not in (None, 0):
        ytdlp_stderr_text = ytdlp_stderr.decode(errors='replace')
        if 'broken pipe' not in ytdlp_stderr_text.lower():
            raise RuntimeError(f'yt-dlp failed: {ytdlp_stderr_text}')

    if ffmpeg_returncode != 0:
        stderr_text = ffmpeg_stderr.decode(errors='replace')
        raise RuntimeError(f'ffmpeg failed: {stderr_text}')

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
    timeout: timedelta = timedelta(seconds=30),
) -> str | None:
    proc = await asyncio.create_subprocess_exec(
        'yt-dlp',
        '--no-warnings',
        '--print',
        '%(acodec)s',
        '-f',
        'bestaudio',
        '--skip-download',
        '--no-playlist',
        url,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=timeout.total_seconds(),
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise

    if proc.returncode != 0:
        stderr_text = stderr.decode(errors='replace')
        raise RuntimeError(f'yt-dlp failed: {stderr_text}')

    codec = stdout.decode(errors='replace').strip()
    if not codec:
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
