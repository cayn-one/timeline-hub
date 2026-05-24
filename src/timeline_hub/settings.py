from datetime import timedelta
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from timeline_hub.types import UserId


class S3Settings(BaseModel):
    endpoint_url: str
    region: str
    bucket: str
    access_key_id: str
    secret_access_key: SecretStr


class Settings(BaseModel):
    # Telegram bot
    bot_token: SecretStr
    superuser_ids: set[UserId]
    user_ids: set[UserId]

    # S3-compatible storage
    s3: S3Settings

    # Delay used to batch forwarded messages before responding
    forward_batch_timeout: timedelta = timedelta(seconds=0.25)
    # Padding line width in space units (1 unit ≈ width of one NBSP character)
    message_width: int = 80

    # Lowest year offered for clip store destinations
    min_clip_year: int = 2022
    # Audio normalization (LUFS target and bitrate)
    normalization_loudness: float = -14
    normalization_bitrate: int = 128

    # Will be clipped with precision of frames
    variant_max_duration: timedelta = timedelta(minutes=30)
    slowest_variant_speed: float = 0.5
    # Maximum summed variant payload to attempt `send_media_group` (MiB)
    media_group_max_size: int = 47

    model_config = ConfigDict(
        frozen=True,
    )

    @classmethod
    def load(cls) -> Self:
        # BaseSettings resolves required fields from environment at runtime.
        env_settings = _EnvSettings()  # pyright: ignore[reportCallIssue]

        return cls(
            bot_token=env_settings.bot_token,
            superuser_ids=env_settings.superuser_ids,
            user_ids=env_settings.user_ids,
            s3=env_settings.s3,
        )

    @model_validator(mode='before')
    @classmethod
    def add_superusers_to_users(cls, data: Any) -> Any:
        if isinstance(data, dict) and ('user_ids' in data or 'superuser_ids' in data):
            data['user_ids'] = set(data.get('user_ids', [])) | set(data.get('superuser_ids', []))
        return data

    @model_validator(mode='after')
    def validate_variant_limits(self) -> Self:
        if self.slowest_variant_speed <= 0 or self.slowest_variant_speed > 1:
            raise ValueError('slowest_variant_speed must satisfy 0 < slowest_variant_speed <= 1')
        if self.media_group_max_size <= 0:
            raise ValueError('media_group_max_size must be > 0')
        return self


class _EnvSettings(BaseSettings):
    bot_token: SecretStr
    superuser_ids: set[UserId]
    user_ids: set[UserId] = Field(default_factory=set)
    s3: S3Settings

    model_config = SettingsConfigDict(
        env_file='.env',
        frozen=True,
        extra='ignore',
        env_nested_delimiter='__',
    )
