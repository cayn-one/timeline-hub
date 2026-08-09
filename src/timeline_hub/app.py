import argparse
import asyncio
import logging
import sys

from aiogram import Bot, Dispatcher
from aiogram.fsm.storage.memory import MemoryStorage
from async_s3 import S3Client, S3Config
from loguru import logger

from timeline_hub.constants import TELEGRAM_MEDIA_GROUP_MAX_ITEMS
from timeline_hub.handlers.router import router as handlers_router
from timeline_hub.infra.tasks import TaskFailure, TaskScheduler, TaskSupervisor
from timeline_hub.middleware import AllowlistMiddleware, TelegramRetryAfterMiddleware
from timeline_hub.services.clip_store import ClipStore
from timeline_hub.services.container import Services
from timeline_hub.services.message_buffer import ChatMessageBuffer
from timeline_hub.services.track_store import Preset, PresetMode, PresetStore, TrackStore
from timeline_hub.services.youtube_cookies import YoutubeCookieStore
from timeline_hub.settings import Settings
from timeline_hub.types import UserId


def main() -> None:
    _configure_logging()
    args = _parse_args()
    settings = Settings.load(is_dev=args.dev)
    asyncio.run(_main(settings))


async def _main(settings: Settings) -> None:
    dp = Dispatcher(storage=MemoryStorage())
    bot = Bot(settings.bot_token.get_secret_value())
    bot.session.middleware.register(TelegramRetryAfterMiddleware())

    async with (
        bot,
        S3Client(
            S3Config(
                endpoint_url=settings.s3.endpoint_url,
                region=settings.s3.region,
                bucket=settings.s3.bucket,
                access_key_id=settings.s3.access_key_id,
                secret_access_key=settings.s3.secret_access_key.get_secret_value(),
            )
        ) as s3_client,
    ):

        async def on_failure_stop(_: TaskFailure | None = None) -> None:
            await _notify_superusers_and_stop_polling(
                bot=bot,
                dispatcher=dp,
                superuser_ids=settings.superuser_ids,
            )

        preset_store = PresetStore(
            s3_client,
            bootstrap_preset=_default_track_preset(),
            namespace=settings.track_namespace,
        )
        youtube_cookie_store = YoutubeCookieStore(s3_client)
        await youtube_cookie_store.refresh()

        dp['services'] = Services(
            chat_message_buffer=ChatMessageBuffer(),
            task_scheduler=TaskScheduler(
                task_supervisor=TaskSupervisor(on_failure=on_failure_stop),
            ),
            clip_store=ClipStore(
                s3_client,
                namespace=settings.clip_namespace,
                max_s3_concurrency=settings.max_s3_concurrency,
                sampled_phash_mean_threshold=settings.sampled_phash_mean_threshold,
                inbox_batch_size=TELEGRAM_MEDIA_GROUP_MAX_ITEMS,
            ),
            track_store=TrackStore(
                s3_client,
                preset_store=preset_store,
                variant_max_duration=settings.variant_max_duration,
                namespace=settings.track_namespace,
            ),
            youtube_cookie_store=youtube_cookie_store,
        )
        dp['settings'] = settings
        dp['on_failure'] = on_failure_stop
        dp.include_router(handlers_router)
        dp.update.middleware(AllowlistMiddleware(user_ids=settings.user_ids))

        logger.info('bot started')
        await dp.start_polling(bot, polling_timeout=30)
        logger.info('bot stopped')


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
                logger.exception('failed to notify superuser {} about shutdown', superuser_id)
    finally:
        await dispatcher.stop_polling()


def _configure_logging() -> None:
    logger.remove()
    logger.add(
        sys.stdout,
        format='{level.name:.2} {time:HH:mm:ss} {message}',
        enqueue=True,
        backtrace=False,
        diagnose=False,
    )
    # Hide normal 'SIGINT` signal logs when shutting bot down
    logging.getLogger('aiogram').setLevel(logging.ERROR)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog='timeline-hub')
    parser.add_argument(
        '--dev',
        action='store_true',
        help='load development overrides from config.toml',
    )
    return parser.parse_args()


def _default_track_preset() -> Preset:
    return Preset(
        name='Default',
        slowed=PresetMode(step=0.08, levels=3),
        sped_up=PresetMode(step=0.06, levels=3),
        reverb_start=0.01,
        reverb_step=0.01,
    )


if __name__ == '__main__':
    main()
