from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call

import pytest
from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest, TelegramRetryAfter
from aiogram.methods import SendMessage

import timeline_hub.app as app_module
from timeline_hub.middleware import TelegramRetryAfterMiddleware


def _method() -> SendMessage:
    return SendMessage(chat_id=123, text='hello')


@pytest.mark.asyncio
async def test_retry_middleware_passes_through_without_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    middleware = TelegramRetryAfterMiddleware()
    make_request = AsyncMock(return_value='ok')
    sleep = AsyncMock()
    monkeypatch.setattr(app_module.asyncio, 'sleep', sleep)

    result = await middleware(make_request, Mock(spec=Bot), _method())

    assert result == 'ok'
    make_request.assert_awaited_once()
    sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_retry_middleware_retries_once_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    middleware = TelegramRetryAfterMiddleware()
    method = _method()
    make_request = AsyncMock(
        side_effect=[
            TelegramRetryAfter(method=method, message='Flood control exceeded', retry_after=2),
            'ok',
        ],
    )
    sleep = AsyncMock()
    monkeypatch.setattr(app_module.asyncio, 'sleep', sleep)

    result = await middleware(make_request, Mock(spec=Bot), method)

    assert result == 'ok'
    assert make_request.await_count == 2
    sleep.assert_awaited_once_with(2.25)


@pytest.mark.asyncio
async def test_retry_budget_is_per_request(monkeypatch: pytest.MonkeyPatch) -> None:
    middleware = TelegramRetryAfterMiddleware()
    sleep = AsyncMock()
    monkeypatch.setattr(app_module.asyncio, 'sleep', sleep)

    first_method = _method()
    first_make_request = AsyncMock(
        side_effect=[
            TelegramRetryAfter(method=first_method, message='Flood control exceeded', retry_after=1),
            'first-ok',
        ],
    )
    second_method = _method()
    second_make_request = AsyncMock(return_value='second-ok')

    first_result = await middleware(first_make_request, Mock(spec=Bot), first_method)
    second_result = await middleware(second_make_request, Mock(spec=Bot), second_method)

    assert first_result == 'first-ok'
    assert second_result == 'second-ok'
    assert first_make_request.await_count == 2
    second_make_request.assert_awaited_once()
    sleep.assert_awaited_once_with(1.25)


@pytest.mark.asyncio
async def test_retry_middleware_exhausts_after_three_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    middleware = TelegramRetryAfterMiddleware()
    method = _method()
    make_request = AsyncMock(
        side_effect=[
            TelegramRetryAfter(method=method, message='Flood control exceeded', retry_after=1),
            TelegramRetryAfter(method=method, message='Flood control exceeded', retry_after=1),
            TelegramRetryAfter(method=method, message='Flood control exceeded', retry_after=1),
            TelegramRetryAfter(method=method, message='Flood control exceeded', retry_after=1),
        ],
    )
    sleep = AsyncMock()
    monkeypatch.setattr(app_module.asyncio, 'sleep', sleep)

    with pytest.raises(TelegramRetryAfter):
        await middleware(make_request, Mock(spec=Bot), method)

    assert make_request.await_count == 4
    assert sleep.await_count == 3
    sleep.assert_has_awaits([call(1.25)] * 3)


@pytest.mark.asyncio
async def test_retry_middleware_re_raises_non_retry_error_immediately(monkeypatch: pytest.MonkeyPatch) -> None:
    middleware = TelegramRetryAfterMiddleware()
    method = _method()
    make_request = AsyncMock(side_effect=TelegramBadRequest(method=method, message='bad request'))
    sleep = AsyncMock()
    monkeypatch.setattr(app_module.asyncio, 'sleep', sleep)

    with pytest.raises(TelegramBadRequest):
        await middleware(make_request, Mock(spec=Bot), method)

    make_request.assert_awaited_once()
    sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_app_registers_retry_middleware_on_bot_session(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeSessionMiddleware:
        def __init__(self) -> None:
            self.registered: list[object] = []

        def register(self, middleware: object) -> object:
            self.registered.append(middleware)
            return middleware

    class _FakeSession:
        def __init__(self) -> None:
            self.middleware = _FakeSessionMiddleware()

    class _FakeBot:
        def __init__(self, token: str) -> None:
            self.token = token
            self.session = _FakeSession()
            created_bots.append(self)

        async def __aenter__(self) -> _FakeBot:
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

    class _FakeS3Client:
        def __init__(self, config) -> None:
            self.config = config

        async def __aenter__(self) -> _FakeS3Client:
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

    class _FakeDispatcher:
        def __init__(self, storage) -> None:
            self.storage = storage
            self.update = SimpleNamespace(middleware=Mock())
            self.items: dict[str, object] = {}
            self.included_routers: list[object] = []
            self.start_polling = AsyncMock(return_value=None)

        def __setitem__(self, key: str, value: object) -> None:
            self.items[key] = value

        def include_router(self, router: object) -> None:
            self.included_routers.append(router)

    settings = SimpleNamespace(
        bot_token=SimpleNamespace(get_secret_value=lambda: '123456:ABCDEF'),
        superuser_ids={1},
        user_ids={1},
        s3=SimpleNamespace(
            endpoint_url='https://s3.local',
            region='us-test-1',
            bucket='bucket',
            access_key_id='ak',
            secret_access_key=SimpleNamespace(get_secret_value=lambda: 'sk'),
        ),
        clip_namespace='clips',
        track_namespace='tracks',
        variant_max_duration=timedelta(minutes=15),
    )
    created_bots: list[object] = []

    monkeypatch.setattr(app_module, 'Bot', _FakeBot)
    monkeypatch.setattr(app_module, 'S3Client', _FakeS3Client)
    monkeypatch.setattr(app_module, 'Dispatcher', _FakeDispatcher)

    await app_module._main(settings)

    assert len(created_bots) == 1
    assert len(created_bots[0].session.middleware.registered) == 1
    assert isinstance(created_bots[0].session.middleware.registered[0], TelegramRetryAfterMiddleware)
