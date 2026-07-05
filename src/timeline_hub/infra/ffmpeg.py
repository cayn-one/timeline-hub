import asyncio
import hashlib
import json
import math
import os
import tempfile
import time
from datetime import timedelta
from pathlib import Path
from statistics import median
from typing import Any, Literal

from PIL import Image, ImageOps

_HASH_READ_SIZE = 64 * 1024
_SUPPORTED_VIDEO_CODECS = ('h264', 'hevc')
SAMPLED_PHASH_SAMPLE_COUNT = 25
_PHASH_IMAGE_SIZE = 32
_PHASH_DCT_BLOCK_SIZE = 8
_PHASH_MAX_VALUE = 2**63


def _dct_matrix(size: int) -> list[list[float]]:
    matrix: list[list[float]] = []
    factor = math.pi / (2.0 * size)
    for u in range(size):
        row = [math.cos((2 * x + 1) * u * factor) for x in range(size)]
        matrix.append(row)
    return matrix


_DCT_32 = _dct_matrix(_PHASH_IMAGE_SIZE)


class UnsupportedVideoCodecError(ValueError):
    """Raised when clip hashing encounters a supported container with an unsupported video codec."""

    def __init__(self, *, codec: str, supported_codecs: tuple[str, ...]) -> None:
        self.codec = codec
        self.supported_codecs = supported_codecs
        super().__init__(f'unsupported video codec: {codec!r}; supported codecs: {supported_codecs}')


class PerceptualMetadataUnavailableError(RuntimeError):
    """Raised when perceptual video metadata cannot be computed reliably."""


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


async def _probe_primary_video_frame_count(
    input_path: Path,
    *,
    timeout: timedelta,
) -> int:
    frame_count_text = await _run_ffprobe_text(
        input_path,
        timeout,
        select_streams='v:0',
        show_entries='stream=nb_frames',
    )
    if not frame_count_text:
        raise PerceptualMetadataUnavailableError('ffprobe returned no primary video frame count')
    try:
        frame_count = int(frame_count_text)
    except ValueError as error:
        raise PerceptualMetadataUnavailableError(
            f'ffprobe returned invalid primary video frame count: {frame_count_text!r}'
        ) from error
    if frame_count <= 0:
        raise PerceptualMetadataUnavailableError(
            f'ffprobe returned non-positive primary video frame count: {frame_count}'
        )
    return frame_count


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


async def compute_video_perceptual_metadata(
    video_bytes: bytes,
    *,
    timeout: timedelta = timedelta(seconds=30),
) -> tuple[int, tuple[int, ...]]:
    """Return sampled perceptual metadata for video deduplication.

    Raises:
        PerceptualMetadataUnavailableError: If reliable frame-count or sampled
            frame decoding cannot be obtained for perceptual comparison.
    """
    frame_count = await compute_video_frame_count(video_bytes, timeout=timeout)
    sampled_phashes = await compute_video_sampled_phashes(video_bytes, frame_count=frame_count, timeout=timeout)
    return frame_count, sampled_phashes


async def compute_video_frame_count(
    video_bytes: bytes,
    *,
    timeout: timedelta = timedelta(seconds=30),
) -> int:
    """Return the primary video stream frame count for perceptual deduplication."""
    input_fd, input_name = tempfile.mkstemp(suffix='.mp4')
    os.close(input_fd)
    input_path = Path(input_name)

    try:
        input_path.write_bytes(video_bytes)
        deadline = time.monotonic() + timeout.total_seconds()
        try:
            return await _probe_primary_video_frame_count(
                input_path,
                timeout=_remaining_timeout(deadline),
            )
        except PerceptualMetadataUnavailableError:
            raise
        except (RuntimeError, asyncio.TimeoutError) as error:
            raise PerceptualMetadataUnavailableError('ffprobe failed to provide perceptual metadata') from error
    finally:
        input_path.unlink(missing_ok=True)


async def compute_video_sampled_phashes(
    video_bytes: bytes,
    *,
    frame_count: int,
    timeout: timedelta = timedelta(seconds=30),
) -> tuple[int, ...]:
    """Return sampled perceptual hashes for a known positive frame count."""
    if frame_count <= 0:
        raise ValueError('frame_count must be >= 1')

    input_fd, input_name = tempfile.mkstemp(suffix='.mp4')
    os.close(input_fd)
    input_path = Path(input_name)

    try:
        input_path.write_bytes(video_bytes)
        deadline = time.monotonic() + timeout.total_seconds()
        frame_indices = _sample_frame_indices(frame_count, sample_count=SAMPLED_PHASH_SAMPLE_COUNT)
        sampled_phashes: list[int] = []
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            frame_paths = await _extract_video_frames(
                input_path,
                frame_indices=frame_indices,
                output_dir=temp_dir_path,
                timeout=_remaining_timeout(deadline),
            )
            for frame_index, frame_path in zip(frame_indices, frame_paths, strict=True):
                try:
                    with Image.open(frame_path) as image:
                        sampled_phashes.append(_perceptual_hash(image.convert('RGB')))
                except OSError as error:
                    raise PerceptualMetadataUnavailableError(f'failed to decode sampled frame {frame_index}') from error

        if not sampled_phashes:
            raise PerceptualMetadataUnavailableError('ffmpeg produced no sampled perceptual frames')

        return tuple(sampled_phashes)
    finally:
        input_path.unlink(missing_ok=True)


def sampled_phash_mean_distance(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    """Return the mean Hamming distance across sampled perceptual hashes."""
    if not left or not right:
        raise ValueError('sampled pHash sequences must be non-empty')
    if len(left) != len(right):
        raise ValueError('sampled pHash sequences must have equal length')
    return sum((left_hash ^ right_hash).bit_count() for left_hash, right_hash in zip(left, right, strict=True)) / len(
        left
    )


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


def _sample_frame_indices(frame_count: int, *, sample_count: int) -> tuple[int, ...]:
    if frame_count <= 0:
        return ()
    if frame_count <= sample_count:
        return tuple(range(frame_count))
    step = (frame_count - 1) / (sample_count - 1)
    return tuple(sorted({round(index * step) for index in range(sample_count)}))


async def _extract_video_frames(
    input_path: Path,
    *,
    frame_indices: tuple[int, ...],
    output_dir: Path,
    timeout: timedelta,
) -> tuple[Path, ...]:
    if not frame_indices:
        raise PerceptualMetadataUnavailableError('ffmpeg received no sampled perceptual frames to extract')

    output_pattern = output_dir / 'frame-%03d.png'
    select_terms = '+'.join(f'eq(n\\,{frame_index})' for frame_index in frame_indices)
    try:
        await _run_ffmpeg(
            (
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
                '-vf',
                f'select={select_terms}',
                '-vsync',
                '0',
                str(output_pattern),
            ),
            timeout,
            capture='none',
        )
    except (RuntimeError, asyncio.TimeoutError) as error:
        raise PerceptualMetadataUnavailableError('ffmpeg failed to extract sampled perceptual frames') from error

    frame_paths = tuple(sorted(output_dir.glob('frame-*.png')))
    if len(frame_paths) != len(frame_indices):
        raise PerceptualMetadataUnavailableError(
            f'ffmpeg extracted {len(frame_paths)} sampled perceptual frames, expected {len(frame_indices)}'
        )
    return frame_paths


def _perceptual_hash(image: Image.Image, *, size: int = _PHASH_IMAGE_SIZE, block: int = _PHASH_DCT_BLOCK_SIZE) -> int:
    gray = ImageOps.grayscale(image).resize((size, size), Image.Resampling.LANCZOS)
    pixels = list(gray.tobytes())
    rows = [pixels[index : index + size] for index in range(0, len(pixels), size)]

    coeffs: list[list[float]] = [[0.0 for _ in range(block)] for _ in range(block)]
    scale = [1.0 / math.sqrt(2.0)] + [1.0 for _ in range(size - 1)]
    for u in range(block):
        for v in range(block):
            total = 0.0
            row_cos = _DCT_32[u]
            col_cos = _DCT_32[v]
            for x in range(size):
                row = rows[x]
                row_cos_x = row_cos[x]
                for y in range(size):
                    total += row[y] * row_cos_x * col_cos[y]
            coeffs[u][v] = (2.0 / size) * scale[u] * scale[v] * total

    values = [coeffs[u][v] for u in range(block) for v in range(block) if not (u == 0 and v == 0)]
    threshold = median(values)
    value = 0
    for coefficient in values:
        value <<= 1
        if coefficient >= threshold:
            value |= 1
    return value
