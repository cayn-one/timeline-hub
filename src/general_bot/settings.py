from datetime import timedelta
from typing import Any, Self

from pydantic import BaseModel, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from general_bot.types import UserId


class _BotTokenSettings(BaseSettings):
    bot_token: SecretStr
    bot_token_dev: SecretStr | None = None

    model_config = SettingsConfigDict(
        env_file='.env',
        frozen=True,
        extra='ignore',
    )


class S3Settings(BaseModel):
    endpoint_url: str
    region: str
    bucket: str
    access_key_id: str
    secret_access_key: SecretStr


class Settings(BaseSettings):
    # Telegram bot
    bot_token: SecretStr
    superuser_ids: set[UserId]
    user_ids: set[UserId]

    # S3-compatible storage
    s3: S3Settings

    # Delay used to batch forwarded messages before responding
    forward_batch_timeout: timedelta = timedelta(seconds=0.25)

    # Audio normalization (LUFS target and bitrate)
    normalization_loudness: float = -14
    normalization_bitrate: int = 128

    model_config = SettingsConfigDict(
        env_file='.env',
        frozen=True,
        extra='ignore',
        env_nested_delimiter='__',
    )

    @classmethod
    def load(cls, is_dev: bool) -> Self:
        bot_token_settings = _BotTokenSettings()
        if is_dev and bot_token_settings.bot_token_dev is None:
            raise ValueError('`BOT_TOKEN_DEV` is required in `.env` in dev mode')
        return cls(
            bot_token=bot_token_settings.bot_token_dev if is_dev else bot_token_settings.bot_token,
        )  # type: ignore[call-arg]  # pydantic-settings fills remaining fields from env at runtime; static checker false positive

    @model_validator(mode='before')
    @classmethod
    def add_superusers_to_users(cls, data: Any) -> Any:
        if isinstance(data, dict) and ('user_ids' in data or 'superuser_ids' in data):
            data['user_ids'] = set(data.get('user_ids', [])) | set(data.get('superuser_ids', []))
        return data
