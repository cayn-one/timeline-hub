from datetime import timedelta
from typing import Any, Self

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from general_bot.types import UserId

_CONFIG = SettingsConfigDict(
    env_file='.env',
    frozen=True,
    extra='ignore',
)


class _BotTokenSettings(BaseSettings):
    bot_token: str
    bot_token_dev: str | None = None

    model_config = _CONFIG


class Settings(BaseSettings):
    # Telegram Bot API token used to authenticate the bot with Telegram
    bot_token: str

    # Telegram user IDs with elevated privileges
    superuser_ids: set[UserId]

    # Telegram user IDs allowed to interact with the bot. Includes superusers
    user_ids: set[UserId]

    # Delay used to batch forwarded messages before responding
    forward_batch_timeout: timedelta = timedelta(seconds=0.25)

    # Target loudness for normalized clips (LUFS)
    normalization_loudness: float = -14

    # Output bitrate for normalized clips (kbps)
    normalization_bitrate: int = 128

    model_config = _CONFIG

    @classmethod
    def load(cls, is_dev: bool) -> Self:
        bts = _BotTokenSettings()
        if is_dev and bts.bot_token_dev is None:
            raise ValueError('`BOT_TOKEN_DEV` is required in `.env` in dev mode')
        return cls(
            bot_token=bts.bot_token_dev if is_dev else bts.bot_token,
        )  # type: ignore[call-arg]  # pydantic-settings fills remaining fields from env at runtime; static checker false positive

    @model_validator(mode='before')
    @classmethod
    def add_superusers_to_users(cls, data: Any) -> Any:
        if isinstance(data, dict) and ('user_ids' in data or 'superuser_ids' in data):
            data['user_ids'] = set(data.get('user_ids', [])) | set(data.get('superuser_ids', []))
        return data
