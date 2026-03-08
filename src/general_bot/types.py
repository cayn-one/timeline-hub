from collections.abc import Awaitable
from typing import Any, Callable

from aiogram.types import TelegramObject

type MiddlewareData = dict[str, Any]
type Handler = Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]]

type ChatId = int
type UserId = int
