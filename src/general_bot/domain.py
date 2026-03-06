import asyncio
import json
import os
import tempfile
from datetime import timedelta
from pathlib import Path


async def normalize_video_volume(
    video_bytes: bytes,
    *,
    loudness: float = -14.0,
    timeout: timedelta = timedelta(seconds=30),
) -> bytes:
    """Normalize video audio volume with 2-pass `loudnorm`.

    The original video stream is copied unchanged, while the audio stream is
    normalized and re-encoded.

    Temporary files are used instead of piping MP4 bytes through ffmpeg
    stdin/stdout because MP4 muxing requires a seekable output.

    Args:
        video_bytes: Original MP4 video bytes.
        loudness: Target integrated loudness in LUFS.
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

        analysis_filter = (
            f'loudnorm=I={loudness}:TP=-1.5:LRA=7:print_format=json'
        )
        analysis_cmd = (
            'ffmpeg',
            '-hide_banner',
            '-loglevel', 'info',
            '-nostats',
            '-nostdin',
            '-y',
            '-threads', '1',
            '-i', str(input_path),
            '-vn',
            '-af', analysis_filter,
            '-f', 'null',
            '-',
        )
        analysis_proc = await asyncio.create_subprocess_exec(
            *analysis_cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            _, analysis_stderr = await asyncio.wait_for(
                analysis_proc.communicate(),
                timeout=timeout.total_seconds(),
            )
        except asyncio.TimeoutError:
            analysis_proc.kill()
            await analysis_proc.wait()
            raise

        if analysis_proc.returncode != 0:
            stderr_text = analysis_stderr.decode(errors='replace')
            raise RuntimeError(f'ffmpeg analysis failed: {stderr_text}')

        analysis_text = analysis_stderr.decode(errors='replace')
        json_start = analysis_text.rfind('{')
        json_end = analysis_text.rfind('}')

        if json_start == -1 or json_end == -1 or json_end < json_start:
            raise RuntimeError(f'ffmpeg analysis output did not contain loudnorm JSON: {analysis_text}')

        stats = json.loads(analysis_text[json_start:json_end + 1])

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
            '-loglevel', 'error',
            '-nostats',
            '-nostdin',
            '-y',
            '-threads', '1',
            '-i', str(input_path),
            '-c:v', 'copy',
            '-af', normalize_filter,
            '-c:a', 'aac',
            '-b:a', '128k',
            str(output_path),
        )
        normalize_proc = await asyncio.create_subprocess_exec(
            *normalize_cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            _, normalize_stderr = await asyncio.wait_for(
                normalize_proc.communicate(),
                timeout=timeout.total_seconds(),
            )
        except asyncio.TimeoutError:
            normalize_proc.kill()
            await normalize_proc.wait()
            raise

        if normalize_proc.returncode != 0:
            stderr_text = normalize_stderr.decode(errors='replace')
            raise RuntimeError(f'ffmpeg normalization failed: {stderr_text}')

        return output_path.read_bytes()

    finally:
        input_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
