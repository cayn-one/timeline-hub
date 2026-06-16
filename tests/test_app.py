from unittest.mock import AsyncMock, Mock, patch

import pytest

import timeline_hub.app as app_module
from timeline_hub.app import _notify_superusers_and_stop_polling


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
