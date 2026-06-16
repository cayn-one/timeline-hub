import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from aiogram import BaseMiddleware, Bot
from aiogram.client.session.middlewares.base import BaseRequestMiddleware, NextRequestMiddlewareType
from aiogram.exceptions import TelegramRetryAfter
from aiogram.methods import TelegramMethod
from aiogram.methods.base import TelegramType
from aiogram.types import TelegramObject, User
from loguru import logger

from timeline_hub.types import UserId


class AllowlistMiddleware(BaseMiddleware):
    def __init__(self, *, user_ids: set[UserId]) -> None:
        self._user_ids = user_ids

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        user: User | None = data.get('event_from_user')
        if user is None:
            return None
        if user.id not in self._user_ids:
            logger.info(
                'user {} (@{} {!r}) attempting to use bot',
                user.id,
                user.username or '',
                user.full_name,
            )
            return None
        return await handler(event, data)


class TelegramRetryAfterMiddleware(BaseRequestMiddleware):
    """Retry Telegram flood-control responses for one outgoing request sequence.

    The retry budget is per outbound Bot API call. Each new request gets a fresh
    retry counter, so the middleware only retries a single request sequence and
    never applies a global budget across the process.
    """

    def __init__(self, *, max_retries: int = 3, retry_buffer_seconds: float = 0.25) -> None:
        if max_retries < 1:
            raise ValueError('`max_retries` must be >= 1')
        if retry_buffer_seconds < 0:
            raise ValueError('`retry_buffer_seconds` must be >= 0')

        self._max_retries = max_retries
        self._retry_buffer_seconds = retry_buffer_seconds

    async def __call__(
        self,
        make_request: NextRequestMiddlewareType[TelegramType],
        bot: Bot,
        method: TelegramMethod[Any],
    ) -> Any:
        retries = 0

        while True:
            try:
                return await make_request(bot, method)
            except TelegramRetryAfter as exc:
                if retries >= self._max_retries:
                    logger.error(
                        'telegram flood control exhausted for {} after {} retries (retry_after={}, max_retries={})',
                        type(method).__name__,
                        self._max_retries,
                        exc.retry_after,
                        self._max_retries,
                    )
                    raise

                retries += 1
                delay = exc.retry_after + self._retry_buffer_seconds
                logger.warning(
                    'telegram flood control hit for {} (retry_after={}, delay={:.2f}, retry={}/{})',
                    type(method).__name__,
                    exc.retry_after,
                    delay,
                    retries,
                    self._max_retries,
                )
                await asyncio.sleep(delay)
