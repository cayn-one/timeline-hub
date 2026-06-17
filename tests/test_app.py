from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from pydantic import SecretStr

import timeline_hub.app as app_module
from timeline_hub.app import _notify_superusers_and_stop_polling
from timeline_hub.settings import S3Settings, Settings


def test_main_loads_production_settings_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = object()
    load = Mock(return_value=settings)
    main = AsyncMock(return_value=None)

    monkeypatch.setattr(app_module, '_configure_logging', Mock())
    monkeypatch.setattr(app_module.Settings, 'load', load)
    monkeypatch.setattr(app_module, '_main', main)
    monkeypatch.setattr(app_module.sys, 'argv', ['timeline-hub'])

    app_module.main()

    load.assert_called_once_with(is_dev=False)
    main.assert_awaited_once_with(settings)


def test_main_loads_dev_overrides_when_flag_is_present(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = object()
    load = Mock(return_value=settings)
    main = AsyncMock(return_value=None)

    monkeypatch.setattr(app_module, '_configure_logging', Mock())
    monkeypatch.setattr(app_module.Settings, 'load', load)
    monkeypatch.setattr(app_module, '_main', main)
    monkeypatch.setattr(app_module.sys, 'argv', ['timeline-hub', '--dev'])

    app_module.main()

    load.assert_called_once_with(is_dev=True)
    main.assert_awaited_once_with(settings)


@pytest.mark.asyncio
async def test_main_wires_storage_namespaces_into_stores(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(
        bot_token=SecretStr('bot-token'),
        superuser_ids={1},
        user_ids={1},
        s3=S3Settings(
            endpoint_url='http://localhost:9000',
            region='us-east-1',
            bucket='bucket',
            access_key_id='key',
            secret_access_key=SecretStr('secret'),
        ),
        forward_batch_timeout=timedelta(milliseconds=250),
        message_width=80,
        min_year=2022,
        clip_namespace='clips-dev',
        normalization_loudness=-14,
        normalization_bitrate=128,
        track_namespace='tracks-dev',
        variant_max_duration=timedelta(minutes=30),
        slowest_variant_speed=0.5,
        media_group_max_size=47,
    )
    dispatcher = MagicMock()
    dispatcher.include_router = Mock()
    dispatcher.update = Mock(middleware=Mock(register=Mock()))
    dispatcher.start_polling = AsyncMock(return_value=None)
    bot = MagicMock()
    bot.session = Mock(middleware=Mock(register=Mock()))

    async def _bot_aenter() -> MagicMock:
        return bot

    async def _bot_aexit(exc_type, exc, tb) -> None:
        return None

    bot.__aenter__.side_effect = _bot_aenter
    bot.__aexit__.side_effect = _bot_aexit
    s3_client = object()
    clip_store = object()
    track_store = object()
    preset_store = object()
    clip_store_ctor = Mock(return_value=clip_store)
    track_store_ctor = Mock(return_value=track_store)
    preset_store_ctor = Mock(return_value=preset_store)

    class _FakeS3ClientContext:
        async def __aenter__(self) -> object:
            return s3_client

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

    monkeypatch.setattr(app_module, 'Dispatcher', Mock(return_value=dispatcher))
    monkeypatch.setattr(app_module, 'Bot', Mock(return_value=bot))
    monkeypatch.setattr(app_module, 'S3Client', Mock(return_value=_FakeS3ClientContext()))
    monkeypatch.setattr(app_module, 'ClipStore', clip_store_ctor)
    monkeypatch.setattr(app_module, 'TrackStore', track_store_ctor)
    monkeypatch.setattr(app_module, 'PresetStore', preset_store_ctor)

    await app_module._main(settings)

    clip_store_ctor.assert_called_once_with(s3_client, namespace='clips-dev')
    preset_store_ctor.assert_called_once()
    assert preset_store_ctor.call_args.args == (s3_client,)
    assert preset_store_ctor.call_args.kwargs['namespace'] == 'tracks-dev'
    track_store_ctor.assert_called_once()
    assert track_store_ctor.call_args.args == (s3_client,)
    assert track_store_ctor.call_args.kwargs['preset_store'] is preset_store
    assert track_store_ctor.call_args.kwargs['namespace'] == 'tracks-dev'


@pytest.mark.asyncio
async def test_failure_shutdown_stops_polling_when_notifications_succeed() -> None:
    bot = AsyncMock()
    dispatcher = AsyncMock()

    with patch('timeline_hub.app.logger.exception'):
        await _notify_superusers_and_stop_polling(
            bot=bot,
            dispatcher=dispatcher,
            superuser_ids={1, 2},
        )

    assert bot.send_message.await_count == 2
    assert {call.kwargs['chat_id'] for call in bot.send_message.await_args_list} == {1, 2}
    dispatcher.stop_polling.assert_awaited_once()


@pytest.mark.asyncio
async def test_failure_shutdown_stops_polling_when_one_notification_fails() -> None:
    bot = AsyncMock()
    bot.send_message.side_effect = [RuntimeError('blocked'), None]
    dispatcher = AsyncMock()

    with patch('timeline_hub.app.logger.exception'):
        await _notify_superusers_and_stop_polling(
            bot=bot,
            dispatcher=dispatcher,
            superuser_ids={1, 2},
        )

    assert bot.send_message.await_count == 2
    assert {call.kwargs['chat_id'] for call in bot.send_message.await_args_list} == {1, 2}
    dispatcher.stop_polling.assert_awaited_once()


@pytest.mark.asyncio
async def test_failure_shutdown_stops_polling_when_all_notifications_fail() -> None:
    bot = AsyncMock()
    bot.send_message.side_effect = [RuntimeError('first'), RuntimeError('second')]
    dispatcher = AsyncMock()

    with patch('timeline_hub.app.logger.exception'):
        await _notify_superusers_and_stop_polling(
            bot=bot,
            dispatcher=dispatcher,
            superuser_ids={1, 2},
        )

    assert bot.send_message.await_count == 2
    assert {call.kwargs['chat_id'] for call in bot.send_message.await_args_list} == {1, 2}
    dispatcher.stop_polling.assert_awaited_once()
