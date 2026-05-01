import asyncio
import re
from datetime import timedelta
from pathlib import Path

import pytest

from timeline_hub.infra import ytdlp as ytdlp_module


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

    assert result == expected_bytes
    args = observed['args']
    assert args[0] == 'yt-dlp'
    assert '-f' in args
    assert 'bestaudio[acodec=opus]/bestaudio' in args
    assert '--extract-audio' in args
    assert '--audio-format' in args
    assert 'opus' in args
    assert '--quiet' in args
    assert '--no-playlist' in args
    assert '-o' in args
    assert args[-1] == 'https://example.com/watch?v=abc'


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
