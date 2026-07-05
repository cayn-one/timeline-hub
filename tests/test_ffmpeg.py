import asyncio
import math
import wave
from datetime import timedelta
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest

from timeline_hub.infra import ffmpeg as ffmpeg_module

_VIDEO_FIXTURE_DIR = Path(__file__).parent / 'fixtures' / 'video'


def _video_fixture(name: str) -> bytes:
    return (_VIDEO_FIXTURE_DIR / name).read_bytes()


@pytest.mark.asyncio
async def test_create_audio_variant_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match='audio_bytes must not be empty'):
        await ffmpeg_module.create_audio_variant(
            b'',
            speed=1.0,
            reverb=0.0,
            input_sample_rate=48_000,
        )


@pytest.mark.asyncio
async def test_create_audio_variant_rejects_invalid_input_sample_rate() -> None:
    with pytest.raises(ValueError, match='input_sample_rate must be >= 1'):
        await ffmpeg_module.create_audio_variant(
            b'source-audio',
            speed=1.0,
            reverb=0.0,
            input_sample_rate=0,
        )


@pytest.mark.asyncio
async def test_create_audio_variant_rejects_non_integer_mp3_quality() -> None:
    with pytest.raises(ValueError, match='mp3_quality must be an integer'):
        await ffmpeg_module.create_audio_variant(
            b'source-audio',
            speed=1.0,
            reverb=0.0,
            input_sample_rate=48_000,
            output_format='mp3',
            mp3_quality=True,
        )


@pytest.mark.asyncio
async def test_create_audio_variant_rejects_out_of_range_mp3_quality() -> None:
    with pytest.raises(ValueError, match='mp3_quality must be in 0..9'):
        await ffmpeg_module.create_audio_variant(
            b'source-audio',
            speed=1.0,
            reverb=0.0,
            input_sample_rate=48_000,
            output_format='mp3',
            mp3_quality=10,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize('max_input_duration', [True, 10])
async def test_create_audio_variant_rejects_invalid_max_input_duration(
    max_input_duration: object,
) -> None:
    with pytest.raises(ValueError, match='max_input_duration must be a timedelta'):
        await ffmpeg_module.create_audio_variant(
            b'source-audio',
            speed=1.0,
            reverb=0.0,
            input_sample_rate=48_000,
            max_input_duration=max_input_duration,  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
@pytest.mark.parametrize('max_input_duration', [timedelta(0), timedelta(seconds=-1)])
async def test_create_audio_variant_rejects_non_positive_max_input_duration(
    max_input_duration: timedelta,
) -> None:
    with pytest.raises(ValueError, match='max_input_duration must be > 0'):
        await ffmpeg_module.create_audio_variant(
            b'source-audio',
            speed=1.0,
            reverb=0.0,
            input_sample_rate=48_000,
            max_input_duration=max_input_duration,
        )


@pytest.mark.asyncio
async def test_create_audio_variant_builds_slowdown_filter_without_reverb(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    async def _fake_run_ffmpeg(
        cmd: tuple[str, ...],
        timeout: timedelta,
        *,
        stdin_bytes: bytes | None = None,
        capture: str = 'none',
    ) -> bytes:
        observed['cmd'] = cmd
        observed['run_timeout'] = timeout
        observed['stdin_bytes'] = stdin_bytes
        observed['capture'] = capture
        return b'variant-audio'

    monkeypatch.setattr(ffmpeg_module, '_run_ffmpeg', _fake_run_ffmpeg)

    result = await ffmpeg_module.create_audio_variant(
        b'source-audio',
        speed=0.75,
        reverb=0.0,
        input_sample_rate=44_100,
        output_format='opus',
        opus_bitrate=96,
        timeout=timedelta(seconds=12),
    )

    assert result == b'variant-audio'
    assert observed['run_timeout'] == timedelta(seconds=12)
    assert observed['stdin_bytes'] == b'source-audio'
    assert observed['capture'] == 'stdout'
    assert observed['cmd'] == (
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
        '-af',
        'asetrate=44100*0.75,aresample=48000:resampler=soxr:precision=28:cheby=1,'
        'equalizer=f=5000:t=q:w=1:g=1,equalizer=f=14000:t=q:w=1:g=-2',
        '-ar',
        '48000',
        '-c:a',
        'libopus',
        '-b:a',
        '96k',
        '-vbr',
        'on',
        '-compression_level',
        '10',
        '-f',
        'opus',
        'pipe:1',
    )


@pytest.mark.asyncio
async def test_create_audio_variant_builds_input_duration_args_before_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    async def _fake_run_ffmpeg(
        cmd: tuple[str, ...],
        timeout: timedelta,
        *,
        stdin_bytes: bytes | None = None,
        capture: str = 'none',
    ) -> bytes:
        observed['cmd'] = cmd
        observed['run_timeout'] = timeout
        observed['stdin_bytes'] = stdin_bytes
        observed['capture'] = capture
        return b'variant-audio'

    monkeypatch.setattr(ffmpeg_module, '_run_ffmpeg', _fake_run_ffmpeg)

    result = await ffmpeg_module.create_audio_variant(
        b'source-audio',
        speed=0.75,
        reverb=0.0,
        input_sample_rate=44_100,
        max_input_duration=timedelta(seconds=90),
        output_format='mp3',
        timeout=timedelta(seconds=12),
    )

    assert result == b'variant-audio'
    assert observed['run_timeout'] == timedelta(seconds=12)
    assert observed['stdin_bytes'] == b'source-audio'
    assert observed['capture'] == 'stdout'
    assert observed['cmd'] == (
        'ffmpeg',
        '-hide_banner',
        '-loglevel',
        'error',
        '-nostats',
        '-nostdin',
        '-y',
        '-threads',
        '1',
        '-t',
        '90.0',
        '-i',
        'pipe:0',
        '-vn',
        '-af',
        'asetrate=44100*0.75,aresample=48000:resampler=soxr:precision=28:cheby=1,'
        'equalizer=f=5000:t=q:w=1:g=1,equalizer=f=14000:t=q:w=1:g=-2',
        '-ar',
        '48000',
        '-c:a',
        'libmp3lame',
        '-q:a',
        '1',
        '-f',
        'mp3',
        'pipe:1',
    )


@pytest.mark.asyncio
async def test_create_audio_variant_builds_speedup_filter_with_reverb(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_cmds: list[tuple[str, ...]] = []

    async def _fake_run_ffmpeg(
        cmd: tuple[str, ...],
        timeout: timedelta,
        *,
        stdin_bytes: bytes | None = None,
        capture: str = 'none',
    ) -> bytes:
        observed_cmds.append(cmd)
        assert timeout == timedelta(seconds=5)
        assert stdin_bytes == b'source-audio'
        assert capture == 'stdout'
        return b'variant-audio'

    monkeypatch.setattr(ffmpeg_module, '_run_ffmpeg', _fake_run_ffmpeg)

    result = await ffmpeg_module.create_audio_variant(
        b'source-audio',
        speed=1.25,
        reverb=0.4,
        input_sample_rate=48_000,
        output_format='opus',
        timeout=timedelta(seconds=5),
    )

    assert result == b'variant-audio'
    assert observed_cmds == [
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
            'pipe:0',
            '-vn',
            '-af',
            'asetrate=48000*1.25,aresample=48000:resampler=soxr:precision=28:cheby=1,'
            'equalizer=f=5000:t=q:w=1:g=2,equalizer=f=14000:t=q:w=1:g=-2,'
            'volume=1.125,alimiter=limit=0.98,aecho=1.0:0.95:50:0.8',
            '-ar',
            '48000',
            '-c:a',
            'libopus',
            '-b:a',
            '160k',
            '-vbr',
            'on',
            '-compression_level',
            '10',
            '-f',
            'opus',
            'pipe:1',
        )
    ]


@pytest.mark.asyncio
async def test_create_audio_variant_builds_mp3_output_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    async def _fake_run_ffmpeg(
        cmd: tuple[str, ...],
        timeout: timedelta,
        *,
        stdin_bytes: bytes | None = None,
        capture: str = 'none',
    ) -> bytes:
        observed['cmd'] = cmd
        observed['run_timeout'] = timeout
        observed['stdin_bytes'] = stdin_bytes
        observed['capture'] = capture
        return b'variant-audio'

    monkeypatch.setattr(ffmpeg_module, '_run_ffmpeg', _fake_run_ffmpeg)

    result = await ffmpeg_module.create_audio_variant(
        b'source-audio',
        speed=1.0,
        reverb=0.0,
        input_sample_rate=48_000,
        output_format='mp3',
        timeout=timedelta(seconds=7),
    )

    assert result == b'variant-audio'
    assert observed['run_timeout'] == timedelta(seconds=7)
    assert observed['stdin_bytes'] == b'source-audio'
    assert observed['capture'] == 'stdout'
    assert observed['cmd'] == (
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
        '-af',
        'asetrate=48000*1.0,aresample=48000:resampler=soxr:precision=28:cheby=1,'
        'equalizer=f=5000:t=q:w=1:g=2,equalizer=f=14000:t=q:w=1:g=-2,'
        'volume=1.0,alimiter=limit=0.98',
        '-ar',
        '48000',
        '-c:a',
        'libmp3lame',
        '-q:a',
        '1',
        '-f',
        'mp3',
        'pipe:1',
    )


@pytest.mark.asyncio
async def test_create_audio_variant_builds_mp3_output_args_with_custom_quality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    async def _fake_run_ffmpeg(
        cmd: tuple[str, ...],
        timeout: timedelta,
        *,
        stdin_bytes: bytes | None = None,
        capture: str = 'none',
    ) -> bytes:
        observed['cmd'] = cmd
        observed['run_timeout'] = timeout
        observed['stdin_bytes'] = stdin_bytes
        observed['capture'] = capture
        return b'variant-audio'

    monkeypatch.setattr(ffmpeg_module, '_run_ffmpeg', _fake_run_ffmpeg)

    result = await ffmpeg_module.create_audio_variant(
        b'source-audio',
        speed=1.0,
        reverb=0.0,
        input_sample_rate=48_000,
        output_format='mp3',
        mp3_quality=4,
        timeout=timedelta(seconds=7),
    )

    assert result == b'variant-audio'
    assert observed['run_timeout'] == timedelta(seconds=7)
    assert observed['stdin_bytes'] == b'source-audio'
    assert observed['capture'] == 'stdout'
    assert observed['cmd'] == (
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
        '-af',
        'asetrate=48000*1.0,aresample=48000:resampler=soxr:precision=28:cheby=1,'
        'equalizer=f=5000:t=q:w=1:g=2,equalizer=f=14000:t=q:w=1:g=-2,'
        'volume=1.0,alimiter=limit=0.98',
        '-ar',
        '48000',
        '-c:a',
        'libmp3lame',
        '-q:a',
        '4',
        '-f',
        'mp3',
        'pipe:1',
    )


@pytest.mark.asyncio
async def test_create_audio_variant_pipe_execution_returns_non_empty_output() -> None:
    audio_bytes = _build_wav_bytes(sample_rate=44_100)

    variant_bytes = await ffmpeg_module.create_audio_variant(
        audio_bytes,
        speed=1.0,
        reverb=0.0,
        input_sample_rate=44_100,
        output_format='opus',
    )

    assert variant_bytes
    assert variant_bytes.startswith(b'OggS')


@pytest.mark.asyncio
async def test_create_audio_variant_pipe_failure_propagates_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_run_ffmpeg(
        cmd: tuple[str, ...],
        timeout: timedelta,
        *,
        stdin_bytes: bytes | None = None,
        capture: str = 'none',
    ) -> bytes:
        raise RuntimeError('ffmpeg failed: broken input')

    monkeypatch.setattr(ffmpeg_module, '_run_ffmpeg', _fake_run_ffmpeg)

    with pytest.raises(RuntimeError, match='ffmpeg failed: broken input'):
        await ffmpeg_module.create_audio_variant(
            b'source-audio',
            speed=1.0,
            reverb=0.0,
            input_sample_rate=48_000,
        )


@pytest.mark.asyncio
async def test_create_audio_variant_pipe_empty_output_raises_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeProc:
        def __init__(self) -> None:
            self.returncode = 0

        async def communicate(self, input_data: bytes | None = None) -> tuple[bytes, bytes]:
            return b'', b''

        def kill(self) -> None:
            return None

        async def wait(self) -> int:
            return self.returncode

    async def _fake_create_subprocess_exec(*args: str, **kwargs: object) -> _FakeProc:
        return _FakeProc()

    monkeypatch.setattr(ffmpeg_module.asyncio, 'create_subprocess_exec', _fake_create_subprocess_exec)

    with pytest.raises(RuntimeError, match='ffmpeg produced empty stdout'):
        await ffmpeg_module.create_audio_variant(
            b'source-audio',
            speed=1.0,
            reverb=0.0,
            input_sample_rate=48_000,
        )


@pytest.mark.asyncio
async def test_run_ffmpeg_capture_none_returns_empty_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeProc:
        returncode = 0

        async def communicate(self, input_data: bytes | None = None) -> tuple[bytes, bytes]:
            return b'', b'info-on-stderr'

        def kill(self) -> None:
            return None

        async def wait(self) -> int:
            return self.returncode

    async def _fake_create_subprocess_exec(*args: str, **kwargs: object) -> _FakeProc:
        return _FakeProc()

    monkeypatch.setattr(ffmpeg_module.asyncio, 'create_subprocess_exec', _fake_create_subprocess_exec)

    result = await ffmpeg_module._run_ffmpeg(('ffmpeg',), timedelta(seconds=1), capture='none')
    assert result == b''


@pytest.mark.asyncio
async def test_run_ffmpeg_capture_stdout_returns_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeProc:
        returncode = 0

        async def communicate(self, input_data: bytes | None = None) -> tuple[bytes, bytes]:
            return b'encoded', b''

        def kill(self) -> None:
            return None

        async def wait(self) -> int:
            return self.returncode

    async def _fake_create_subprocess_exec(*args: str, **kwargs: object) -> _FakeProc:
        return _FakeProc()

    monkeypatch.setattr(ffmpeg_module.asyncio, 'create_subprocess_exec', _fake_create_subprocess_exec)

    result = await ffmpeg_module._run_ffmpeg(('ffmpeg',), timedelta(seconds=1), capture='stdout')
    assert result == b'encoded'


@pytest.mark.asyncio
async def test_run_ffmpeg_capture_stderr_returns_stderr(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeProc:
        returncode = 0

        async def communicate(self, input_data: bytes | None = None) -> tuple[bytes, bytes]:
            return b'', b'analysis-json'

        def kill(self) -> None:
            return None

        async def wait(self) -> int:
            return self.returncode

    async def _fake_create_subprocess_exec(*args: str, **kwargs: object) -> _FakeProc:
        return _FakeProc()

    monkeypatch.setattr(ffmpeg_module.asyncio, 'create_subprocess_exec', _fake_create_subprocess_exec)

    result = await ffmpeg_module._run_ffmpeg(('ffmpeg',), timedelta(seconds=1), capture='stderr')
    assert result == b'analysis-json'


@pytest.mark.asyncio
async def test_normalize_video_audio_loudness_uses_stderr_capture_for_analysis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    loudnorm_json = (
        '{"input_i":"-20.1","input_tp":"-1.0","input_lra":"3.0","input_thresh":"-30.0","target_offset":"0.5"}'
    )

    async def _fake_run_ffmpeg(
        cmd: tuple[str, ...],
        timeout: timedelta,
        *,
        stdin_bytes: bytes | None = None,
        capture: str = 'none',
    ) -> bytes:
        calls.append({'cmd': cmd, 'capture': capture, 'timeout': timeout, 'stdin_bytes': stdin_bytes})
        if len(calls) == 1:
            return loudnorm_json.encode()
        return b''

    monkeypatch.setattr(ffmpeg_module, '_run_ffmpeg', _fake_run_ffmpeg)

    class _FakePath:
        def __init__(self, name: str) -> None:
            self._name = name

        def write_bytes(self, data: bytes) -> int:
            return len(data)

        def read_bytes(self) -> bytes:
            return b'normalized-video'

        def unlink(self, missing_ok: bool = False) -> None:
            return None

        def __str__(self) -> str:
            return self._name

    monkeypatch.setattr(ffmpeg_module, 'Path', _FakePath)
    monkeypatch.setattr(ffmpeg_module.tempfile, 'mkstemp', lambda suffix='': (0, f'/tmp/fake{suffix}'))
    monkeypatch.setattr(ffmpeg_module.os, 'close', lambda fd: None)

    result = await ffmpeg_module.normalize_video_audio_loudness(b'video-bytes')

    assert result == b'normalized-video'
    assert calls[0]['capture'] == 'stderr'
    assert calls[1]['capture'] == 'none'


@pytest.mark.asyncio
async def test_hash_video_content_preserves_h264_digest_regression() -> None:
    assert await ffmpeg_module.hash_video_content(_video_fixture('h264.mp4')) == (
        '8c332ee605067eca7bf986fb82ca2957dc49467cf3311ca62257ff31f28e23ae'
    )


@pytest.mark.asyncio
async def test_hash_video_content_supports_hevc() -> None:
    assert await ffmpeg_module.hash_video_content(_video_fixture('hevc.mp4')) == (
        'e49208a15f6b1cd4aa4e779d97ed966fe77715cd6b06570c1de7bb03f2ff6b73'
    )


@pytest.mark.asyncio
async def test_hash_video_content_shares_one_total_timeout_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, timedelta] = {}
    monotonic_values = iter([100.0, 103.0, 107.0])

    async def _fake_probe(input_path: Path, *, timeout: timedelta) -> str:
        del input_path
        observed['probe_timeout'] = timeout
        return 'h264'

    async def _fake_hash(cmd: tuple[str, ...], timeout: timedelta) -> str:
        del cmd
        observed['hash_timeout'] = timeout
        return 'fake-digest'

    monkeypatch.setattr(ffmpeg_module, 'time', SimpleNamespace(monotonic=lambda: next(monotonic_values)))
    monkeypatch.setattr(ffmpeg_module, '_probe_primary_video_codec', _fake_probe)
    monkeypatch.setattr(ffmpeg_module, '_hash_process_stdout', _fake_hash)

    assert await ffmpeg_module.hash_video_content(b'clip-bytes', timeout=timedelta(seconds=10)) == ('fake-digest')
    assert observed['probe_timeout'] == timedelta(seconds=7)
    assert observed['hash_timeout'] == timedelta(seconds=3)


@pytest.mark.asyncio
async def test_hash_video_content_raises_for_unsupported_codec(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fake_probe(input_path: Path, *, timeout: timedelta) -> str:
        del input_path, timeout
        return 'vp9'

    monkeypatch.setattr(ffmpeg_module, '_probe_primary_video_codec', _fake_probe)

    with pytest.raises(ffmpeg_module.UnsupportedVideoCodecError) as excinfo:
        await ffmpeg_module.hash_video_content(_video_fixture('h264.mp4'))

    assert excinfo.value.codec == 'vp9'
    assert excinfo.value.supported_codecs == ('h264', 'hevc')


@pytest.mark.asyncio
async def test_compute_video_frame_count_returns_frame_count() -> None:
    assert await ffmpeg_module.compute_video_frame_count(_video_fixture('h264.mp4')) == 1


@pytest.mark.asyncio
async def test_compute_video_sampled_phashes_returns_hashes() -> None:
    assert await ffmpeg_module.compute_video_sampled_phashes(_video_fixture('h264.mp4'), frame_count=1) == (
        6846486214506547647,
    )


@pytest.mark.asyncio
async def test_compute_video_perceptual_metadata_returns_frame_count_and_hashes() -> None:
    frame_count, sampled_phashes = await ffmpeg_module.compute_video_perceptual_metadata(_video_fixture('h264.mp4'))

    assert frame_count == 1
    assert len(sampled_phashes) == 1
    assert sampled_phashes == (6846486214506547647,)


@pytest.mark.asyncio
async def test_compute_video_perceptual_metadata_raises_unavailable_when_frame_count_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_probe(input_path: Path, *, timeout: timedelta) -> int:
        del input_path, timeout
        raise ffmpeg_module.PerceptualMetadataUnavailableError('ffprobe returned no primary video frame count')

    monkeypatch.setattr(ffmpeg_module, '_probe_primary_video_frame_count', _fake_probe)

    with pytest.raises(ffmpeg_module.PerceptualMetadataUnavailableError, match='no primary video frame count'):
        await ffmpeg_module.compute_video_perceptual_metadata(_video_fixture('h264.mp4'))


@pytest.mark.asyncio
async def test_compute_video_perceptual_metadata_raises_unavailable_when_frame_probe_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_probe(input_path: Path, *, timeout: timedelta) -> int:
        del input_path, timeout
        raise RuntimeError('ffprobe failed')

    monkeypatch.setattr(ffmpeg_module, '_probe_primary_video_frame_count', _fake_probe)

    with pytest.raises(
        ffmpeg_module.PerceptualMetadataUnavailableError,
        match='ffprobe failed to provide perceptual metadata',
    ):
        await ffmpeg_module.compute_video_perceptual_metadata(_video_fixture('h264.mp4'))


@pytest.mark.asyncio
async def test_compute_video_sampled_phashes_rejects_non_positive_frame_count() -> None:
    with pytest.raises(ValueError, match='frame_count must be >= 1'):
        await ffmpeg_module.compute_video_sampled_phashes(_video_fixture('h264.mp4'), frame_count=0)


@pytest.mark.asyncio
async def test_compute_video_perceptual_metadata_extracts_sampled_frames_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[Path, tuple[int, ...], Path, timedelta]] = []

    async def _fake_probe(input_path: Path, *, timeout: timedelta) -> int:
        del input_path, timeout
        return 3

    async def _fake_extract(
        input_path: Path,
        *,
        frame_indices: tuple[int, ...],
        output_dir: Path,
        timeout: timedelta,
    ) -> tuple[Path, ...]:
        calls.append((input_path, frame_indices, output_dir, timeout))
        return (tmp_path / 'frame-001.png', tmp_path / 'frame-002.png', tmp_path / 'frame-003.png')

    class _FakeImage:
        def convert(self, mode: str) -> object:
            del mode
            return object()

    class _FakeImageContext:
        def __enter__(self) -> _FakeImage:
            return _FakeImage()

        def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
            del exc_type, exc, traceback
            return None

    phashes = iter((11, 22, 33))

    monkeypatch.setattr(ffmpeg_module, '_probe_primary_video_frame_count', _fake_probe)
    monkeypatch.setattr(ffmpeg_module, '_extract_video_frames', _fake_extract)
    monkeypatch.setattr(ffmpeg_module.Image, 'open', lambda path: _FakeImageContext())
    monkeypatch.setattr(ffmpeg_module, '_perceptual_hash', lambda image: next(phashes))

    frame_count, sampled_phashes = await ffmpeg_module.compute_video_perceptual_metadata(_video_fixture('h264.mp4'))

    assert frame_count == 3
    assert sampled_phashes == (11, 22, 33)
    assert len(calls) == 1
    assert calls[0][1] == (0, 1, 2)


@pytest.mark.asyncio
async def test_compute_video_perceptual_metadata_raises_unavailable_when_sampled_frame_decode_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_decode_error(*args, **kwargs):
        del args, kwargs
        raise OSError('cannot decode frame')

    monkeypatch.setattr(ffmpeg_module.Image, 'open', _raise_decode_error)

    with pytest.raises(ffmpeg_module.PerceptualMetadataUnavailableError, match='failed to decode sampled frame'):
        await ffmpeg_module.compute_video_perceptual_metadata(_video_fixture('h264.mp4'))


@pytest.mark.asyncio
async def test_compute_video_perceptual_metadata_raises_unavailable_when_frame_extract_times_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_extract(
        input_path: Path,
        *,
        frame_indices: tuple[int, ...],
        output_dir: Path,
        timeout: timedelta,
    ) -> tuple[Path, ...]:
        del input_path, frame_indices, output_dir, timeout
        raise ffmpeg_module.PerceptualMetadataUnavailableError('ffmpeg failed to extract sampled perceptual frames')

    monkeypatch.setattr(ffmpeg_module, '_extract_video_frames', _fake_extract)

    with pytest.raises(
        ffmpeg_module.PerceptualMetadataUnavailableError,
        match='failed to extract sampled perceptual frames',
    ):
        await ffmpeg_module.compute_video_perceptual_metadata(_video_fixture('h264.mp4'))


def test_sampled_phash_mean_distance_returns_zero_for_identical_sequences() -> None:
    assert ffmpeg_module.sampled_phash_mean_distance((0b1010, 0b0101), (0b1010, 0b0101)) == 0.0


def test_sampled_phash_mean_distance_returns_fractional_mean() -> None:
    assert ffmpeg_module.sampled_phash_mean_distance((0b0000, 0b1111), (0b0011, 0b1100)) == 2.0


def test_sampled_phash_mean_distance_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match='equal length'):
        ffmpeg_module.sampled_phash_mean_distance((1,), (1, 2))


@pytest.mark.asyncio
async def test_probe_audio_sample_rate_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match='audio_bytes must not be empty'):
        await ffmpeg_module.probe_audio_sample_rate(b'')


@pytest.mark.asyncio
async def test_to_opus_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match='audio_bytes must not be empty'):
        await ffmpeg_module.to_opus(b'')


@pytest.mark.asyncio
async def test_to_opus_rejects_invalid_bitrate() -> None:
    with pytest.raises(ValueError, match='bitrate must be >= 1'):
        await ffmpeg_module.to_opus(b'source-audio', bitrate=0)


@pytest.mark.asyncio
async def test_to_opus_converts_audio_to_non_empty_output() -> None:
    audio_bytes = _build_wav_bytes(sample_rate=44_100)

    opus_bytes = await ffmpeg_module.to_opus(audio_bytes)

    assert opus_bytes
    assert opus_bytes.startswith(b'OggS')


@pytest.mark.asyncio
async def test_to_opus_outputs_48khz_audio() -> None:
    audio_bytes = _build_wav_bytes(sample_rate=44_100)

    opus_bytes = await ffmpeg_module.to_opus(audio_bytes)

    assert await ffmpeg_module.probe_audio_sample_rate(opus_bytes) == 48_000


@pytest.mark.asyncio
async def test_to_opus_uses_pipe_based_ffmpeg_and_returns_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    async def _fake_run_ffmpeg(
        cmd: tuple[str, ...],
        timeout: timedelta,
        *,
        stdin_bytes: bytes | None = None,
        capture: str = 'none',
    ) -> bytes:
        observed['cmd'] = cmd
        observed['timeout'] = timeout
        observed['stdin_bytes'] = stdin_bytes
        observed['capture'] = capture
        return b'OggSfake-opus-data'

    def _unexpected_mkstemp(*args: object, **kwargs: object) -> tuple[int, str]:
        raise AssertionError('tempfile.mkstemp should not be used by to_opus')

    monkeypatch.setattr(ffmpeg_module, '_run_ffmpeg', _fake_run_ffmpeg)
    monkeypatch.setattr(ffmpeg_module.tempfile, 'mkstemp', _unexpected_mkstemp)

    result = await ffmpeg_module.to_opus(
        b'source-audio',
        bitrate=96,
        timeout=timedelta(seconds=9),
    )

    assert result == b'OggSfake-opus-data'
    assert observed['timeout'] == timedelta(seconds=9)
    assert observed['stdin_bytes'] == b'source-audio'
    assert observed['capture'] == 'stdout'
    assert observed['cmd'] == (
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
        '96k',
        '-vbr',
        'on',
        '-compression_level',
        '10',
        '-f',
        'opus',
        'pipe:1',
    )


@pytest.mark.asyncio
async def test_to_opus_rejects_non_ogg_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_run_ffmpeg(
        cmd: tuple[str, ...],
        timeout: timedelta,
        *,
        stdin_bytes: bytes | None = None,
        capture: str = 'none',
    ) -> bytes:
        return b'not-ogg'

    monkeypatch.setattr(ffmpeg_module, '_run_ffmpeg', _fake_run_ffmpeg)

    with pytest.raises(RuntimeError, match='ffmpeg output is not a valid Ogg/Opus container'):
        await ffmpeg_module.to_opus(b'source-audio')


@pytest.mark.asyncio
async def test_clip_mp3_returns_clipped_mp3_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, object] = {}

    async def _fake_run_ffmpeg(
        cmd: tuple[str, ...],
        timeout: timedelta,
        *,
        stdin_bytes: bytes | None = None,
        capture: str = 'none',
    ) -> bytes:
        observed['cmd'] = cmd
        observed['timeout'] = timeout
        observed['stdin_bytes'] = stdin_bytes
        observed['capture'] = capture
        Path(cmd[-1]).write_bytes(b'ID3\x04\x00\x00\x00\x00\x00\x21fake-mp3')
        return b''

    monkeypatch.setattr(ffmpeg_module, '_run_ffmpeg', _fake_run_ffmpeg)

    result = await ffmpeg_module.clip_mp3(
        b'ID3source-mp3',
        max_duration=timedelta(seconds=90),
        timeout=timedelta(seconds=11),
    )

    assert result.startswith(b'ID3')
    assert observed['timeout'] == timedelta(seconds=11)
    assert observed['stdin_bytes'] == b'ID3source-mp3'
    assert observed['capture'] == 'none'
    observed_cmd = observed['cmd']
    assert isinstance(observed_cmd, tuple)
    assert observed_cmd[:-1] == (
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
        '90.0',
        '-vn',
        '-c:a',
        'copy',
    )
    assert observed_cmd[-1].endswith('.mp3')


@pytest.mark.asyncio
async def test_clip_mp3_raises_when_ffmpeg_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fake_run_ffmpeg(
        cmd: tuple[str, ...],
        timeout: timedelta,
        *,
        stdin_bytes: bytes | None = None,
        capture: str = 'none',
    ) -> bytes:
        raise RuntimeError('ffmpeg failed: invalid input')

    monkeypatch.setattr(ffmpeg_module, '_run_ffmpeg', _fake_run_ffmpeg)

    with pytest.raises(RuntimeError, match='ffmpeg failed: invalid input'):
        await ffmpeg_module.clip_mp3(
            b'ID3source-mp3',
            max_duration=timedelta(seconds=10),
        )


@pytest.mark.asyncio
async def test_clip_mp3_timeout_kills_process_and_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    observed = {'killed': False, 'waited': False}

    class _FakeProc:
        returncode = None

        async def communicate(self, input_data: bytes | None = None) -> tuple[bytes, bytes]:
            await asyncio.sleep(1)
            return b'', b''

        def kill(self) -> None:
            observed['killed'] = True

        async def wait(self) -> int:
            observed['waited'] = True
            return 0

    async def _fake_create_subprocess_exec(*args: str, **kwargs: object) -> _FakeProc:
        return _FakeProc()

    monkeypatch.setattr(ffmpeg_module.asyncio, 'create_subprocess_exec', _fake_create_subprocess_exec)

    with pytest.raises(asyncio.TimeoutError):
        await ffmpeg_module.clip_mp3(
            b'ID3source-mp3',
            max_duration=timedelta(seconds=10),
            timeout=timedelta(milliseconds=1),
        )

    assert observed['killed'] is True
    assert observed['waited'] is True


@pytest.mark.asyncio
async def test_clip_mp3_writes_duration_consistent_seekable_mp3(tmp_path: Path) -> None:
    source_mp3 = await ffmpeg_module.create_audio_variant(
        _build_wav_bytes(sample_rate=44_100, duration_seconds=65),
        speed=1.0,
        reverb=0.0,
        input_sample_rate=44_100,
        output_format='mp3',
        timeout=timedelta(seconds=20),
    )

    clipped_mp3 = await ffmpeg_module.clip_mp3(
        source_mp3,
        max_duration=timedelta(seconds=30),
        timeout=timedelta(seconds=20),
    )

    clipped_path = tmp_path / 'clipped.mp3'
    clipped_path.write_bytes(clipped_mp3)
    format_duration = await _probe_format_duration(clipped_path)
    packet_duration = await _probe_last_packet_time(clipped_path)

    assert format_duration == pytest.approx(30, abs=1)
    assert packet_duration == pytest.approx(30, abs=1)
    assert abs(format_duration - packet_duration) <= 1


def _build_wav_bytes(*, sample_rate: int, duration_seconds: float = 0.1) -> bytes:
    frame_count = int(sample_rate * duration_seconds)
    amplitude = 12_000
    frequency_hz = 440.0
    frames = bytearray()

    for index in range(frame_count):
        sample = int(amplitude * math.sin(2 * math.pi * frequency_hz * (index / sample_rate)))
        frames.extend(sample.to_bytes(2, byteorder='little', signed=True))

    output = BytesIO()
    with wave.open(output, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(bytes(frames))

    return output.getvalue()


async def _probe_format_duration(path: Path) -> float:
    stdout = await _run_probe(
        (
            'ffprobe',
            '-v',
            'error',
            '-show_entries',
            'format=duration',
            '-of',
            'default=nokey=1:noprint_wrappers=1',
            str(path),
        )
    )
    return float(stdout.decode().strip())


async def _probe_last_packet_time(path: Path) -> float:
    stdout = await _run_probe(
        (
            'ffprobe',
            '-v',
            'error',
            '-select_streams',
            'a:0',
            '-show_entries',
            'packet=pts_time',
            '-of',
            'csv=p=0',
            str(path),
        )
    )
    last_packet_time = stdout.decode().strip().splitlines()[-1].rstrip(',')
    return float(last_packet_time)


async def _run_probe(cmd: tuple[str, ...]) -> bytes:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timedelta(seconds=10).total_seconds())
    if proc.returncode != 0:
        raise RuntimeError(f'ffprobe failed: {stderr.decode(errors="replace")}')
    return stdout
