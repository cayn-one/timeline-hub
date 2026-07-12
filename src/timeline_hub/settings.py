import math
import tomllib
from datetime import timedelta
from pathlib import Path
from typing import Any, Self, TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from timeline_hub.types import UserId

CONFIG_PATH = Path('config.toml')


class S3Settings(BaseModel):
    endpoint_url: str
    region: str
    bucket: str
    access_key_id: str
    secret_access_key: SecretStr


class _FileInterfaceConfig(BaseModel):
    forward_batch_timeout_ms: int = Field(gt=0)
    message_width: int = Field(gt=0)

    model_config = ConfigDict(
        frozen=True,
        extra='forbid',
    )


class _FileInterfaceOverrides(BaseModel):
    forward_batch_timeout_ms: int | None = Field(default=None, gt=0)
    message_width: int | None = Field(default=None, gt=0)

    model_config = ConfigDict(
        frozen=True,
        extra='forbid',
    )


class _FileTelegramConfig(BaseModel):
    media_group_max_size: int = Field(gt=0)

    model_config = ConfigDict(
        frozen=True,
        extra='forbid',
    )


class _FileTelegramOverrides(BaseModel):
    media_group_max_size: int | None = Field(default=None, gt=0)

    model_config = ConfigDict(
        frozen=True,
        extra='forbid',
    )


class _FileStoresConfig(BaseModel):
    min_year: int = Field(gt=0)

    model_config = ConfigDict(
        frozen=True,
        extra='forbid',
    )


class _FileStoresOverrides(BaseModel):
    min_year: int | None = Field(default=None, gt=0)

    model_config = ConfigDict(
        frozen=True,
        extra='forbid',
    )


class _FileClipsConfig(BaseModel):
    namespace: str = Field(min_length=1)
    normalization_loudness: float
    normalization_bitrate: int = Field(gt=0)
    max_s3_concurrency: int = Field(gt=0)
    sampled_phash_mean_threshold: float = Field(gt=0, le=63)

    model_config = ConfigDict(
        frozen=True,
        extra='forbid',
    )


class _FileClipsOverrides(BaseModel):
    namespace: str | None = Field(default=None, min_length=1)
    normalization_loudness: float | None = None
    normalization_bitrate: int | None = Field(default=None, gt=0)
    max_s3_concurrency: int | None = Field(default=None, gt=0)
    sampled_phash_mean_threshold: float | None = Field(default=None, gt=0, le=63)

    model_config = ConfigDict(
        frozen=True,
        extra='forbid',
    )


class _FileTracksConfig(BaseModel):
    namespace: str = Field(min_length=1)
    variant_max_duration_minutes: int = Field(gt=0)
    slowest_variant_speed: float = Field(gt=0, le=1)

    model_config = ConfigDict(
        frozen=True,
        extra='forbid',
    )


class _FileTracksOverrides(BaseModel):
    namespace: str | None = Field(default=None, min_length=1)
    variant_max_duration_minutes: int | None = Field(default=None, gt=0)
    slowest_variant_speed: float | None = Field(default=None, gt=0, le=1)

    model_config = ConfigDict(
        frozen=True,
        extra='forbid',
    )


class _FileDevConfig(BaseModel):
    interface: _FileInterfaceOverrides | None = None
    telegram: _FileTelegramOverrides | None = None
    stores: _FileStoresOverrides | None = None
    clips: _FileClipsOverrides | None = None
    tracks: _FileTracksOverrides | None = None

    model_config = ConfigDict(
        frozen=True,
        extra='forbid',
    )


class _FileConfig(BaseModel):
    interface: _FileInterfaceConfig
    telegram: _FileTelegramConfig
    stores: _FileStoresConfig
    clips: _FileClipsConfig
    tracks: _FileTracksConfig
    dev: _FileDevConfig | None = None

    model_config = ConfigDict(
        frozen=True,
        extra='forbid',
    )


class Settings(BaseModel):
    # Telegram bot
    bot_token: SecretStr
    superuser_ids: set[UserId]
    user_ids: set[UserId]

    # S3-compatible storage
    s3: S3Settings

    forward_batch_timeout: timedelta
    message_width: int

    min_year: int
    clip_namespace: str
    normalization_loudness: float
    normalization_bitrate: int
    max_s3_concurrency: int
    sampled_phash_mean_threshold: float

    track_namespace: str
    variant_max_duration: timedelta
    slowest_variant_speed: float
    media_group_max_size: int

    model_config = ConfigDict(
        frozen=True,
    )

    @classmethod
    def load(cls, *, is_dev: bool) -> Self:
        # BaseSettings resolves required fields from environment at runtime.
        env_settings = _EnvSettings()  # pyright: ignore[reportCallIssue]
        file_settings = _load_file_settings(is_dev=is_dev)

        return cls(
            bot_token=env_settings.bot_token,
            superuser_ids=env_settings.superuser_ids,
            user_ids=env_settings.user_ids,
            s3=env_settings.s3,
            forward_batch_timeout=timedelta(milliseconds=file_settings.interface.forward_batch_timeout_ms),
            message_width=file_settings.interface.message_width,
            media_group_max_size=file_settings.telegram.media_group_max_size,
            min_year=file_settings.stores.min_year,
            clip_namespace=file_settings.clips.namespace,
            normalization_loudness=file_settings.clips.normalization_loudness,
            normalization_bitrate=file_settings.clips.normalization_bitrate,
            max_s3_concurrency=file_settings.clips.max_s3_concurrency,
            sampled_phash_mean_threshold=file_settings.clips.sampled_phash_mean_threshold,
            track_namespace=file_settings.tracks.namespace,
            variant_max_duration=timedelta(minutes=file_settings.tracks.variant_max_duration_minutes),
            slowest_variant_speed=file_settings.tracks.slowest_variant_speed,
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
        if self.max_s3_concurrency <= 0:
            raise ValueError('max_s3_concurrency must be > 0')
        if not math.isfinite(self.sampled_phash_mean_threshold):
            raise ValueError('sampled_phash_mean_threshold must be finite')
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


_TSection = TypeVar('_TSection', bound=BaseModel)


def _load_file_settings(*, is_dev: bool) -> _FileConfig:
    with CONFIG_PATH.open('rb') as file:
        raw_config = tomllib.load(file)

    file_config = _FileConfig.model_validate(raw_config)
    if not is_dev or file_config.dev is None:
        return file_config

    return _FileConfig(
        interface=_merge_file_section(file_config.interface, file_config.dev.interface),
        telegram=_merge_file_section(file_config.telegram, file_config.dev.telegram),
        stores=_merge_file_section(file_config.stores, file_config.dev.stores),
        clips=_merge_file_section(file_config.clips, file_config.dev.clips),
        tracks=_merge_file_section(file_config.tracks, file_config.dev.tracks),
        dev=file_config.dev,
    )


def _merge_file_section(base: _TSection, override: BaseModel | None) -> _TSection:
    if override is None:
        return base

    section_data = base.model_dump()
    section_data.update(override.model_dump(exclude_unset=True))
    return cast(_TSection, type(base).model_validate(section_data))
