import asyncio
import hashlib
import json
import math
import os
import tempfile
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal

_HASH_READ_SIZE = 64 * 1024
_SUPPORTED_VIDEO_CODECS = ('h264', 'hevc')


class UnsupportedVideoCodecError(ValueError):
    """Raised when clip hashing encounters a supported container with an unsupported video codec."""

    def __init__(self, *, codec: str, supported_codecs: tuple[str, ...]) -> None:
        self.codec = codec
        self.supported_codecs = supported_codecs
        super().__init__(f'unsupported video codec: {codec!r}; supported codecs: {supported_codecs}')


async def to_opus(
    audio_bytes: bytes,
    *,
    bitrate: int = 160,
    timeout: timedelta = timedelta(seconds=30),
) -> bytes:
    """Convert ffmpeg-readable audio bytes to Opus in an Ogg container.

    Args:
        audio_bytes: Source audio bytes in an ffmpeg-readable audio format.
        bitrate: Target Opus bitrate in kbps.
        timeout: Maximum time allowed for the ffmpeg subprocess run.

    Raises:
        ValueError: If parameters are invalid.
        RuntimeError: If ffmpeg fails.
    """
    if not audio_bytes:
        raise ValueError('audio_bytes must not be empty')

    if isinstance(bitrate, bool) or not isinstance(bitrate, int):
        raise ValueError('bitrate must be an integer')
    if bitrate < 1:
        raise ValueError('bitrate must be >= 1')

    cmd = (
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
        '-vn',
        '-ar',
        '48000',
        '-c:a',
        'libopus',
        '-b:a',
        f'{bitrate}k',
        '-vbr',
        'on',
        '-compression_level',
        '10',
        '-f',
        'opus',
        'pipe:1',
    )
    output = await _run_ffmpeg(
        cmd,
        timeout,
        stdin_bytes=audio_bytes,
        capture='stdout',
    )
    if not output.startswith(b'OggS'):
        raise RuntimeError('ffmpeg output is not a valid Ogg/Opus container')
    return output


async def clip_mp3(
    audio: bytes,
    *,
    max_duration: timedelta,
    timeout: timedelta = timedelta(minutes=2),
) -> bytes:
    """Clip MP3 bytes to a maximum duration and finalize seekable metadata.

    Args:
        audio: Source MP3 bytes.
        max_duration: Maximum output duration.
        timeout: Maximum time allowed for the ffmpeg subprocess run.

    Raises:
        ValueError: If parameters are invalid.
        RuntimeError: If ffmpeg fails or output validation fails.
    """
    if not audio:
        raise ValueError('audio must not be empty')

    if isinstance(max_duration, bool) or not isinstance(max_duration, timedelta):
        raise ValueError('max_duration must be a timedelta')
    if max_duration <= timedelta(0):
        raise ValueError('max_duration must be > 0')

    output_fd, output_name = tempfile.mkstemp(suffix='.mp3')
    os.close(output_fd)
    output_path = Path(output_name)

    try:
        cmd = (
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
            str(max_duration.total_seconds()),
            '-vn',
            '-c:a',
            'copy',
            str(output_path),
        )
        await _run_ffmpeg(
            cmd,
            timeout,
            stdin_bytes=audio,
            capture='none',
        )
        output = output_path.read_bytes()
        if not output:
            raise RuntimeError('ffmpeg produced empty MP3 output')
        is_mp3 = output.startswith(b'ID3') or (len(output) >= 2 and output[0] == 0xFF and (output[1] & 0xE0) == 0xE0)
        if not is_mp3:
            raise RuntimeError('ffmpeg output is not a valid MP3 stream')
        return output
    finally:
        output_path.unlink(missing_ok=True)


async def create_audio_variant(
    audio_bytes: bytes,
    *,
    speed: float,
    reverb: float,
    input_sample_rate: int,
    max_input_duration: timedelta | None = None,
    output_format: Literal['opus', 'mp3'] = 'opus',
    opus_bitrate: int = 160,
    mp3_quality: int = 1,
    timeout: timedelta = timedelta(minutes=3),
) -> bytes:
    """Return an audio variant generated from source audio bytes.

    The input is treated as an ffmpeg-readable audio file. Typical caller
    inputs include formats such as `.opus`, `.mp3`, `.wav`, `.m4a`, or other
    common audio formats supported by ffmpeg.

    Output format is selected explicitly via `output_format`:
    - `'opus'` -> Opus in an Ogg container (bitrate-controlled)
    - `'mp3'` -> MP3 encoded with libmp3lame (VBR quality-controlled)

    Variant generation uses the restored baseline wet path:
    - change playback speed by adjusting sample rate, then resample back to
      the target output sample rate with high-quality SOXR resampling
    - apply branch-specific EQ to shape presence and tame upper highs
    - for `speed >= 1`, apply a moderated volume boost and a light limiter
    - if `reverb > 0`, apply scaled echo reverb at the end of the chain
    - if `reverb == 0`, no reverb is applied

    Args:
        audio_bytes: Source audio bytes in an ffmpeg-readable audio format.
        speed: Playback speed multiplier. Must be > 0.
        reverb: Reverb intensity in the closed range 0..1.
        input_sample_rate: Source audio sample rate in Hz.
        max_input_duration: Optional maximum source duration to process before
            applying speed, filtering, and reverb.
        output_format: Target output format. Supported values are `'opus'`
            and `'mp3'`.
        opus_bitrate: Target Opus bitrate in kbps. Used only when
            `output_format='opus'`.
        mp3_quality: MP3 VBR quality level (LAME `-q:a`). Lower is higher
            quality. Typical range is 0-9, with 1 being very high quality.
            Used only when `output_format='mp3'`.
        timeout: Maximum time allowed for the ffmpeg subprocess run.

    Raises:
        ValueError: If parameters are invalid.
        RuntimeError: If ffmpeg fails.
    """
    if not audio_bytes:
        raise ValueError('audio_bytes must not be empty')

    if isinstance(speed, bool) or not isinstance(speed, int | float):
        raise ValueError('speed must be numeric')
    speed = float(speed)
    if not math.isfinite(speed):
        raise ValueError('speed must be finite')
    if speed <= 0:
        raise ValueError('speed must be > 0')

    if isinstance(reverb, bool) or not isinstance(reverb, int | float):
        raise ValueError('reverb must be numeric')
    reverb = float(reverb)
    if not math.isfinite(reverb):
        raise ValueError('reverb must be finite')
    if reverb < 0 or reverb > 1:
        raise ValueError('reverb must be in 0..1')

    if isinstance(input_sample_rate, bool) or not isinstance(input_sample_rate, int):
        raise ValueError('input_sample_rate must be an integer')
    if input_sample_rate < 1:
        raise ValueError('input_sample_rate must be >= 1')

    if output_format not in {'opus', 'mp3'}:
        raise ValueError("output_format must be 'opus' or 'mp3'")

    if isinstance(opus_bitrate, bool) or not isinstance(opus_bitrate, int):
        raise ValueError('opus_bitrate must be an integer')
    if opus_bitrate < 1:
        raise ValueError('opus_bitrate must be >= 1')

    if isinstance(mp3_quality, bool) or not isinstance(mp3_quality, int):
        raise ValueError('mp3_quality must be an integer')
    if not 0 <= mp3_quality <= 9:
        raise ValueError('mp3_quality must be in 0..9')

    input_duration_args: tuple[str, ...] = ()
    if max_input_duration is not None:
        if isinstance(max_input_duration, bool) or not isinstance(max_input_duration, timedelta):
            raise ValueError('max_input_duration must be a timedelta')
        if max_input_duration <= timedelta(0):
            raise ValueError('max_input_duration must be > 0')
        input_duration_args = (
            '-t',
            str(max_input_duration.total_seconds()),
        )

    if output_format == 'opus':
        output_sample_rate = 48_000
        output_muxer = 'opus'
        codec_args = (
            '-c:a',
            'libopus',
            '-b:a',
            f'{opus_bitrate}k',
            '-vbr',
            'on',
            '-compression_level',
            '10',
        )
    else:
        output_sample_rate = 48_000
        output_muxer = 'mp3'
        codec_args = (
            '-c:a',
            'libmp3lame',
            '-q:a',
            str(mp3_quality),
        )
    filter_parts = [
        f'asetrate={input_sample_rate}*{speed}',
        f'aresample={output_sample_rate}:resampler=soxr:precision=28:cheby=1',
    ]

    if speed < 1.0:
        filter_parts.extend(
            [
                'equalizer=f=5000:t=q:w=1:g=1',
                'equalizer=f=14000:t=q:w=1:g=-2',
            ]
        )
    else:
        fast_volume = 1.0 + (speed - 1.0) * 0.5
        filter_parts.extend(
            [
                'equalizer=f=5000:t=q:w=1:g=2',
                'equalizer=f=14000:t=q:w=1:g=-2',
                f'volume={fast_volume}',
                'alimiter=limit=0.98',
            ]
        )

    effective_reverb = reverb * 2.0
    if effective_reverb > 0:
        echo_decay = min(max(effective_reverb, 0.001), 0.98)
        filter_parts.append(f'aecho=1.0:0.95:50:{echo_decay}')

    audio_filter = ','.join(filter_parts)

    cmd = (
        'ffmpeg',
        '-hide_banner',
        '-loglevel',
        'error',
        '-nostats',
        '-nostdin',
        '-y',
        '-threads',
        '1',
        *input_duration_args,
        '-i',
        'pipe:0',
        '-vn',
        '-af',
        audio_filter,
        '-ar',
        str(output_sample_rate),
        *codec_args,
        '-f',
        output_muxer,
        'pipe:1',
    )
    return await _run_ffmpeg(
        cmd,
        timeout,
        stdin_bytes=audio_bytes,
        capture='stdout',
    )


async def probe_audio_sample_rate(
    audio_bytes: bytes,
    *,
    timeout: timedelta = timedelta(seconds=30),
) -> int:
    """Return the sample rate of source audio bytes in Hz.

    The input is treated as an ffmpeg-readable audio file. Typical caller
    inputs include formats such as `.opus`, `.mp3`, `.wav`, `.m4a`, or other
    common audio formats supported by ffprobe.

    Args:
        audio_bytes: Source audio bytes in an ffprobe-readable audio format.
        timeout: Maximum time allowed for the ffprobe subprocess run.

    Raises:
        ValueError: If `audio_bytes` is empty.
        RuntimeError: If ffprobe fails or the sample rate cannot be parsed.
    """
    if not audio_bytes:
        raise ValueError('audio_bytes must not be empty')

    input_fd, input_name = tempfile.mkstemp(suffix='.audio')
    os.close(input_fd)
    input_path = Path(input_name)

    try:
        input_path.write_bytes(audio_bytes)
        sample_rate_text = await _run_ffprobe_text(
            input_path,
            timeout,
            select_streams='a:0',
            show_entries='stream=sample_rate',
        )
        try:
            sample_rate = int(sample_rate_text)
        except ValueError as error:
            raise RuntimeError(f'Failed to parse sample rate: {sample_rate_text!r}') from error

        if sample_rate < 1:
            raise RuntimeError(f'Invalid probed sample rate: {sample_rate}')

        return sample_rate

    finally:
        input_path.unlink(missing_ok=True)


async def _probe_primary_video_codec(
    input_path: Path,
    *,
    timeout: timedelta,
) -> str:
    codec = await _run_ffprobe_text(
        input_path,
        timeout,
        select_streams='v:0',
        show_entries='stream=codec_name',
    )
    if not codec:
        raise RuntimeError('ffprobe returned no primary video codec')

    return codec


async def normalize_video_audio_loudness(
    video_bytes: bytes,
    *,
    loudness: float = -14,
    bitrate: int = 128,
    timeout: timedelta = timedelta(seconds=30),
) -> bytes:
    """Normalize video audio loudness with 2-pass `loudnorm`.

    The original video stream is copied unchanged, while the audio stream is
    normalized and re-encoded.

    Temporary files are used instead of piping MP4 bytes through ffmpeg
    stdin/stdout because MP4 muxing requires a seekable output.

    Args:
        video_bytes: Original MP4 video bytes.
        loudness: Target integrated loudness in LUFS.
        bitrate: Target audio bitrate in kbps for the re-encoded audio stream.
        timeout: Maximum time allowed for each ffmpeg subprocess run.
    """
    input_fd, input_name = tempfile.mkstemp(suffix='.mp4')
    output_fd, output_name = tempfile.mkstemp(suffix='.mp4')
    os.close(input_fd)
    os.close(output_fd)

    input_path = Path(input_name)
    output_path = Path(output_name)

    try:
        input_path.write_bytes(video_bytes)

        analysis_cmd = (
            'ffmpeg',
            '-hide_banner',
            '-loglevel',
            'info',
            '-nostats',
            '-nostdin',
            '-y',
            '-threads',
            '1',
            '-i',
            str(input_path),
            '-vn',
            '-af',
            f'loudnorm=I={loudness}:TP=-1.5:LRA=7:print_format=json',
            '-f',
            'null',
            '-',
        )
        analysis_stderr = await _run_ffmpeg(analysis_cmd, timeout, capture='stderr')

        analysis_text = analysis_stderr.decode(errors='replace')
        json_start = analysis_text.rfind('{')
        json_end = analysis_text.rfind('}')
        if json_start == -1 or json_end == -1 or json_end < json_start:
            raise RuntimeError(f'ffmpeg analysis output did not contain loudnorm JSON: {analysis_text}')
        stats = json.loads(analysis_text[json_start : json_end + 1])

        normalize_filter = (
            f'loudnorm=I={loudness}:TP=-1.5:LRA=7:'
            f'measured_I={stats["input_i"]}:'
            f'measured_TP={stats["input_tp"]}:'
            f'measured_LRA={stats["input_lra"]}:'
            f'measured_thresh={stats["input_thresh"]}:'
            f'offset={stats["target_offset"]}:'
            'linear=true,'
            'alimiter=limit=-1.5dB'
        )
        normalize_cmd = (
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
            str(input_path),
            '-c:v',
            'copy',
            '-af',
            normalize_filter,
            '-c:a',
            'aac',
            '-b:a',
            f'{bitrate}k',
            str(output_path),
        )
        await _run_ffmpeg(normalize_cmd, timeout, capture='none')

        return output_path.read_bytes()

    finally:
        input_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)


async def hash_video_content(
    video_bytes: bytes,
    *,
    timeout: timedelta = timedelta(seconds=30),
) -> str:
    """Return a stable SHA-256 hash of the primary encoded video stream.

    The hash is computed from the first video stream only, after lossless
    extraction into a codec-appropriate elementary stream. Audio, subtitles,
    data streams, and MP4 container metadata are excluded from the hash.
    Original uploaded bytes remain authoritative storage bytes elsewhere; this
    hash is an encoded-stream identity, not a perceptual visual identity.

    Args:
        video_bytes: Original MP4 video bytes.
        timeout: Maximum time allowed for the ffmpeg subprocess run.
    """
    input_fd, input_name = tempfile.mkstemp(suffix='.mp4')
    os.close(input_fd)
    input_path = Path(input_name)

    try:
        input_path.write_bytes(video_bytes)
        deadline = time.monotonic() + timeout.total_seconds()
        codec = await _probe_primary_video_codec(
            input_path,
            timeout=_remaining_timeout(deadline),
        )
        try:
            bitstream_filter, output_format = {
                'h264': ('h264_mp4toannexb', 'h264'),
                'hevc': ('hevc_mp4toannexb', 'hevc'),
            }[codec]
        except KeyError as error:
            raise UnsupportedVideoCodecError(
                codec=codec,
                supported_codecs=_SUPPORTED_VIDEO_CODECS,
            ) from error

        cmd = (
            'ffmpeg',
            '-hide_banner',
            '-loglevel',
            'error',
            '-nostats',
            '-nostdin',
            '-threads',
            '1',
            '-i',
            str(input_path),
            '-map',
            '0:v:0',
            '-c:v',
            'copy',
            '-an',
            '-sn',
            '-dn',
            '-bsf:v',
            bitstream_filter,
            '-f',
            output_format,
            'pipe:1',
        )
        return await _hash_process_stdout(
            cmd,
            _remaining_timeout(deadline),
        )
    finally:
        input_path.unlink(missing_ok=True)


async def _run_ffmpeg(
    cmd: tuple[str, ...],
    timeout: timedelta,
    *,
    stdin_bytes: bytes | None = None,
    capture: Literal['none', 'stdout', 'stderr'] = 'none',
) -> bytes:
    stdout_target = asyncio.subprocess.PIPE if capture == 'stdout' else asyncio.subprocess.DEVNULL
    stdout, stderr = await _run_process(
        cmd,
        timeout,
        stdin_bytes=stdin_bytes,
        stdout=stdout_target,
        error_prefix='ffmpeg failed',
    )
    if capture == 'stderr' and not stderr:
        raise RuntimeError('ffmpeg produced empty stderr')
    if capture == 'stdout' and not stdout:
        raise RuntimeError('ffmpeg produced empty stdout')
    return stderr if capture == 'stderr' else stdout if capture == 'stdout' else b''


async def _run_process(
    cmd: tuple[str, ...],
    timeout: timedelta,
    *,
    stdin_bytes: bytes | None = None,
    stdout: int | None = asyncio.subprocess.DEVNULL,
    error_prefix: str,
) -> tuple[bytes, bytes]:
    input_data = stdin_bytes if stdin_bytes is not None else None
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE if input_data is not None else None,
        stdout=stdout,
        stderr=asyncio.subprocess.PIPE,
    )

    stdout_bytes, stderr_bytes = await _communicate_with_timeout(
        proc,
        timeout,
        input_data=input_data,
    )

    if proc.returncode != 0:
        stderr_text = stderr_bytes.decode(errors='replace')
        raise RuntimeError(f'{error_prefix}: {stderr_text}')

    return stdout_bytes, stderr_bytes


async def _run_ffprobe_text(
    input_path: Path,
    timeout: timedelta,
    *,
    select_streams: str,
    show_entries: str,
) -> str:
    stdout, stderr = await _run_process(
        (
            'ffprobe',
            '-v',
            'error',
            '-select_streams',
            select_streams,
            '-show_entries',
            show_entries,
            '-of',
            'default=nokey=1:noprint_wrappers=1',
            str(input_path),
        ),
        timeout,
        stdout=asyncio.subprocess.PIPE,
        error_prefix='ffprobe failed',
    )
    if not stdout:
        return ''
    return stdout.decode().strip()


async def _hash_process_stdout(
    cmd: tuple[str, ...],
    timeout: timedelta,
) -> str:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    if proc.stdout is None or proc.stderr is None:
        raise RuntimeError('ffmpeg subprocess did not expose stdout/stderr pipes')

    hasher = hashlib.sha256()
    try:
        _, stderr, returncode = await asyncio.wait_for(
            asyncio.gather(
                _hash_stream(proc.stdout, hasher),
                proc.stderr.read(),
                proc.wait(),
            ),
            timeout=timeout.total_seconds(),
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise

    if returncode != 0:
        raise RuntimeError(f'ffmpeg failed while hashing clip: {stderr.decode(errors="replace")}')

    return hasher.hexdigest()


async def _communicate_with_timeout(
    proc: asyncio.subprocess.Process,
    timeout: timedelta,
    *,
    input_data: bytes | None,
) -> tuple[bytes, bytes]:
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(input_data),
            timeout=timeout.total_seconds(),
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise

    return stdout_bytes, stderr_bytes


def _remaining_timeout(deadline: float) -> timedelta:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise asyncio.TimeoutError
    return timedelta(seconds=remaining)


async def _hash_stream(stream: asyncio.StreamReader, hasher: Any) -> None:
    while chunk := await stream.read(_HASH_READ_SIZE):
        hasher.update(chunk)
