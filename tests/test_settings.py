from datetime import timedelta
from pathlib import Path
from textwrap import dedent

import pytest
from pydantic import ValidationError

import timeline_hub.settings as settings_module
from timeline_hub.settings import Settings

_ENV_CONTENT = dedent(
    """\
    BOT_TOKEN=main-token
    SUPERUSER_IDS=[1]
    USER_IDS=[2]

    S3__ENDPOINT_URL=http://localhost:9000
    S3__REGION=us-east-1
    S3__BUCKET=bucket
    S3__ACCESS_KEY_ID=key
    S3__SECRET_ACCESS_KEY=secret
    """
)

_CONFIG_CONTENT = dedent(
    """\
    [interface]
    forward_batch_timeout_ms = 250
    message_width = 80

    [telegram]
    media_group_max_size = 47

    [stores]
    min_year = 2022

    [clips]
    normalization_loudness = -14
    normalization_bitrate = 128

    [tracks]
    variant_max_duration_minutes = 30
    slowest_variant_speed = 0.5
    """
)

_DEV_CONFIG_CONTENT = dedent(
    """\
    [dev.interface]
    forward_batch_timeout_ms = 125
    message_width = 72

    [dev.telegram]
    media_group_max_size = 10

    [dev.clips]
    normalization_loudness = -12
    normalization_bitrate = 96

    [dev.tracks]
    variant_max_duration_minutes = 1
    slowest_variant_speed = 0.75
    """
)


def _write_runtime_files(tmp_path: Path, *, config_content: str = _CONFIG_CONTENT) -> None:
    (tmp_path / '.env').write_text(_ENV_CONTENT)
    (tmp_path / 'config.toml').write_text(config_content)


def _clear_runtime_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        'BOT_TOKEN',
        'SUPERUSER_IDS',
        'USER_IDS',
        'S3__ENDPOINT_URL',
        'S3__REGION',
        'S3__BUCKET',
        'S3__ACCESS_KEY_ID',
        'S3__SECRET_ACCESS_KEY',
    ):
        monkeypatch.delenv(key, raising=False)


def test_settings_load_reads_production_config_and_env_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_runtime_env(monkeypatch)
    _write_runtime_files(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(settings_module, 'CONFIG_PATH', tmp_path / 'config.toml')

    settings = Settings.load(is_dev=False)

    assert settings.bot_token.get_secret_value() == 'main-token'
    assert settings.superuser_ids == {1}
    assert settings.user_ids == {1, 2}
    assert settings.s3.endpoint_url == 'http://localhost:9000'
    assert settings.s3.region == 'us-east-1'
    assert settings.s3.bucket == 'bucket'
    assert settings.s3.access_key_id == 'key'
    assert settings.s3.secret_access_key.get_secret_value() == 'secret'
    assert settings.forward_batch_timeout == timedelta(milliseconds=250)
    assert settings.message_width == 80
    assert settings.media_group_max_size == 47
    assert settings.min_year == 2022
    assert settings.normalization_loudness == -14
    assert settings.normalization_bitrate == 128
    assert settings.variant_max_duration == timedelta(minutes=30)
    assert settings.slowest_variant_speed == 0.5


def test_settings_load_applies_dev_overrides_from_same_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_runtime_env(monkeypatch)
    _write_runtime_files(tmp_path, config_content=_CONFIG_CONTENT + '\n' + _DEV_CONFIG_CONTENT)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(settings_module, 'CONFIG_PATH', tmp_path / 'config.toml')

    settings = Settings.load(is_dev=True)

    assert settings.forward_batch_timeout == timedelta(milliseconds=125)
    assert settings.message_width == 72
    assert settings.media_group_max_size == 10
    assert settings.min_year == 2022
    assert settings.normalization_loudness == -12
    assert settings.normalization_bitrate == 96
    assert settings.variant_max_duration == timedelta(minutes=1)
    assert settings.slowest_variant_speed == 0.75


def test_settings_load_rejects_unknown_config_keys(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_runtime_env(monkeypatch)
    _write_runtime_files(
        tmp_path,
        config_content=dedent(
            """\
            [interface]
            forward_batch_timeout_ms = 250
            message_width = 80
            unexpected = 1

            [telegram]
            media_group_max_size = 47

            [stores]
            min_year = 2022

            [clips]
            normalization_loudness = -14
            normalization_bitrate = 128

            [tracks]
            variant_max_duration_minutes = 30
            slowest_variant_speed = 0.5
            """
        ),
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(settings_module, 'CONFIG_PATH', tmp_path / 'config.toml')

    with pytest.raises(ValidationError):
        Settings.load(is_dev=False)
