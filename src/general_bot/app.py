import argparse
import asyncio
import logging
import sys
from typing import Any

from aiogram import Bot, Dispatcher
from aiogram.types import Update, User
from loguru import logger

from general_bot import handlers
from general_bot.infra.tasks import TaskFailure, TaskScheduler, TaskSupervisor
from general_bot.services import ChatMessageBuffer, Services
from general_bot.settings import Settings
from general_bot.types import Data, Handler, UserId


def run() -> None:
    args = _parse_args()
    settings = Settings.load(args.dev)
    _configure_logging()
    asyncio.run(_main(settings))


async def _main(settings: Settings) -> None:
    dp = Dispatcher()

    async with Bot(settings.bot_token) as bot:
        async def on_failure_stop(_: TaskFailure | None = None) -> None:
            await _notify_superusers_and_stop_polling(
                bot=bot,
                dispatcher=dp,
                superuser_ids=settings.superuser_ids,
            )

        dp['services'] = Services(
            task_scheduler=TaskScheduler(task_supervisor=TaskSupervisor(on_failure=on_failure_stop)),
            chat_message_buffer=ChatMessageBuffer(),
        )
        dp['settings'] = settings
        dp['on_failure'] = on_failure_stop
        dp.include_router(handlers.router)

        @dp.update.middleware()
        async def enforce_allowlist(handler: Handler, update: Update, data: Data) -> Any:
            user: User | None = data.get('event_from_user')
            if user is None:
                return None
            if user.id not in settings.user_ids:
                logger.info(
                    'User {} (@{} {!r}) attempting to use bot',
                    user.id,
                    user.username or '',
                    user.full_name,
                )
                return None
            return await handler(update, data)

        logger.info('Starting bot')
        await dp.start_polling(bot, polling_timeout=30)
        logger.info('Bot stopped')


async def _notify_superusers_and_stop_polling(
    *,
    bot: Bot,
    dispatcher: Dispatcher,
    superuser_ids: set[UserId],
) -> None:
    try:
        for superuser_id in superuser_ids:
            try:
                await bot.send_message(chat_id=superuser_id, text='Stopping bot due to error')
            except Exception:
                logger.exception('Failed to notify superuser {} about shutdown', superuser_id)
    finally:
        await dispatcher.stop_polling()


def _configure_logging() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        format='{message}',
        enqueue=True,
        backtrace=False,
        diagnose=False,
    )
    # Hide normal 'SIGINT` signal logs when shutting bot down
    logging.getLogger('aiogram').setLevel(logging.ERROR)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dev',
        action='store_true',
        help='Run bot in development mode',
    )
    return parser.parse_args()
