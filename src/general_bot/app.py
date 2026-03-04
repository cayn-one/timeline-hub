import asyncio

from aiogram import Bot, Dispatcher

from general_bot import handlers
from general_bot.config import config


def run() -> None:
    asyncio.run(_main())


async def _main() -> None:
    bot = Bot(config.bot_token)
    dp = Dispatcher()
    dp.include_router(handlers.router)

    async with bot:
        await dp.start_polling(bot, polling_timeout=30)
