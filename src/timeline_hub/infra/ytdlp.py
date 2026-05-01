import asyncio
import tempfile
from datetime import timedelta
from pathlib import Path


async def download_audio_as_opus(
    url: str,
    *,
    timeout: timedelta = timedelta(minutes=3),
) -> bytes:
    """Download one URL audio track as Opus bytes using `yt-dlp`.

    Args:
        url: Source URL to download.
        timeout: Maximum time allowed for the `yt-dlp` subprocess run.

    Raises:
        ValueError: If `url` is invalid.
        RuntimeError: If `yt-dlp` fails or output validation fails.
    """
    if not isinstance(url, str):
        raise ValueError('url must be a string')

    normalized_url = url.strip()
    if not normalized_url:
        raise ValueError('url must not be empty')

    with tempfile.TemporaryDirectory() as temp_dir:
        output_template = Path(temp_dir) / 'audio.%(ext)s'
        proc = await asyncio.create_subprocess_exec(
            'yt-dlp',
            '-f',
            'bestaudio[acodec=opus]/bestaudio',
            '--extract-audio',
            '--audio-format',
            'opus',
            '--quiet',
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

        output_files = sorted(Path(temp_dir).glob('*.opus'))
        if not output_files:
            raise RuntimeError('yt-dlp did not produce opus output')
        if len(output_files) > 1:
            raise RuntimeError('yt-dlp produced multiple opus outputs')

        data = output_files[0].read_bytes()
        if not data.startswith(b'OggS'):
            raise RuntimeError('yt-dlp output is not a valid Ogg/Opus container')
        return data
