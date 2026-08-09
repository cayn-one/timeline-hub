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
    media_group_max_size = 49

    [stores]
    min_year = 2022

    [clips]
    namespace = "clips"
    normalization_loudness = -14
    normalization_bitrate = 128
    max_s3_concurrency = 8
    route_store_batch_size = 8
    sampled_phash_mean_threshold = 1.5

    [tracks]
    namespace = "tracks"
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
    namespace = "clips-dev"
    normalization_loudness = -12
    normalization_bitrate = 96
    max_s3_concurrency = 4
    route_store_batch_size = 3
    sampled_phash_mean_threshold = 2.5

    [dev.tracks]
    namespace = "tracks-dev"
    variant_max_duration_minutes = 1
    slowest_variant_speed = 0.75
    """
)


def _write_runtime_files(tmp_path: Path, *, config_content: str = _CONFIG_CONTENT) -> None:
    (tmp_path / '.env').write_text(_ENV_CONTENT)
    (tmp_path / 'config.toml').write_text(config_content)


def _replace_config_line(config_content: str, *, old: str, new: str) -> str:
    return config_content.replace(old, new, 1)


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
    assert settings.media_group_max_size == 49
    assert settings.min_year == 2022
    assert settings.clip_namespace == 'clips'
    assert settings.normalization_loudness == -14
    assert settings.normalization_bitrate == 128
    assert settings.max_s3_concurrency == 8
    assert settings.route_store_batch_size == 8
    assert settings.sampled_phash_mean_threshold == 1.5
    assert settings.track_namespace == 'tracks'
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
    assert settings.clip_namespace == 'clips-dev'
    assert settings.normalization_loudness == -12
    assert settings.normalization_bitrate == 96
    assert settings.max_s3_concurrency == 4
    assert settings.route_store_batch_size == 3
    assert settings.sampled_phash_mean_threshold == 2.5
    assert settings.track_namespace == 'tracks-dev'
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
            media_group_max_size = 49

            [stores]
            min_year = 2022

            [clips]
            namespace = "clips"
            normalization_loudness = -14
            normalization_bitrate = 128
            max_s3_concurrency = 8
            route_store_batch_size = 8
            sampled_phash_mean_threshold = 1.5

            [tracks]
            namespace = "tracks"
            variant_max_duration_minutes = 30
            slowest_variant_speed = 0.5
            """
        ),
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(settings_module, 'CONFIG_PATH', tmp_path / 'config.toml')

    with pytest.raises(ValidationError):
        Settings.load(is_dev=False)


@pytest.mark.parametrize(
    ('old_line', 'new_line'),
    [
        ('namespace = "clips"', 'namespace = ""'),
        ('namespace = "tracks"', 'namespace = ""'),
    ],
)
def test_settings_load_rejects_empty_required_namespaces(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    old_line: str,
    new_line: str,
) -> None:
    _clear_runtime_env(monkeypatch)
    _write_runtime_files(
        tmp_path,
        config_content=_replace_config_line(_CONFIG_CONTENT, old=old_line, new=new_line),
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(settings_module, 'CONFIG_PATH', tmp_path / 'config.toml')

    with pytest.raises(ValidationError):
        Settings.load(is_dev=False)


@pytest.mark.parametrize(
    ('old_line', 'new_line'),
    [
        ('namespace = "clips-dev"', 'namespace = ""'),
        ('namespace = "tracks-dev"', 'namespace = ""'),
    ],
)
def test_settings_load_rejects_empty_dev_namespace_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    old_line: str,
    new_line: str,
) -> None:
    _clear_runtime_env(monkeypatch)
    _write_runtime_files(
        tmp_path,
        config_content=_replace_config_line(
            _CONFIG_CONTENT + '\n' + _DEV_CONFIG_CONTENT,
            old=old_line,
            new=new_line,
        ),
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(settings_module, 'CONFIG_PATH', tmp_path / 'config.toml')

    with pytest.raises(ValidationError):
        Settings.load(is_dev=True)


def test_settings_load_uses_tracked_root_config_for_prod_and_dev_namespaces(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_runtime_env(monkeypatch)
    (tmp_path / '.env').write_text(_ENV_CONTENT)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        settings_module,
        'CONFIG_PATH',
        Path(__file__).resolve().parents[1] / 'config.toml',
    )

    production_settings = Settings.load(is_dev=False)
    development_settings = Settings.load(is_dev=True)

    assert production_settings.clip_namespace == 'clips'
    assert production_settings.track_namespace == 'tracks'
    assert development_settings.clip_namespace == 'clips-dev'
    assert development_settings.track_namespace == 'tracks-dev'


def test_settings_load_rejects_zero_sampled_phash_mean_threshold(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_runtime_env(monkeypatch)
    _write_runtime_files(
        tmp_path,
        config_content=_replace_config_line(
            _CONFIG_CONTENT,
            old='sampled_phash_mean_threshold = 1.5',
            new='sampled_phash_mean_threshold = 0',
        ),
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(settings_module, 'CONFIG_PATH', tmp_path / 'config.toml')

    with pytest.raises(ValidationError):
        Settings.load(is_dev=False)


def test_settings_load_rejects_zero_max_s3_concurrency(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_runtime_env(monkeypatch)
    _write_runtime_files(
        tmp_path,
        config_content=_replace_config_line(
            _CONFIG_CONTENT,
            old='max_s3_concurrency = 8',
            new='max_s3_concurrency = 0',
        ),
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(settings_module, 'CONFIG_PATH', tmp_path / 'config.toml')

    with pytest.raises(ValidationError):
        Settings.load(is_dev=False)


def test_settings_load_rejects_negative_max_s3_concurrency(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_runtime_env(monkeypatch)
    _write_runtime_files(
        tmp_path,
        config_content=_replace_config_line(
            _CONFIG_CONTENT,
            old='max_s3_concurrency = 8',
            new='max_s3_concurrency = -1',
        ),
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(settings_module, 'CONFIG_PATH', tmp_path / 'config.toml')

    with pytest.raises(ValidationError):
        Settings.load(is_dev=False)


@pytest.mark.parametrize('value', [0, -1])
def test_settings_load_rejects_nonpositive_route_store_batch_size(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    value: int,
) -> None:
    _clear_runtime_env(monkeypatch)
    _write_runtime_files(
        tmp_path,
        config_content=_replace_config_line(
            _CONFIG_CONTENT,
            old='route_store_batch_size = 8',
            new=f'route_store_batch_size = {value}',
        ),
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(settings_module, 'CONFIG_PATH', tmp_path / 'config.toml')

    with pytest.raises(ValidationError):
        Settings.load(is_dev=False)


def test_settings_load_accepts_route_store_batch_size_above_telegram_limit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_runtime_env(monkeypatch)
    _write_runtime_files(
        tmp_path,
        config_content=_replace_config_line(
            _CONFIG_CONTENT,
            old='route_store_batch_size = 8',
            new='route_store_batch_size = 12',
        ),
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(settings_module, 'CONFIG_PATH', tmp_path / 'config.toml')

    assert Settings.load(is_dev=False).route_store_batch_size == 12


def test_settings_load_accepts_max_sampled_phash_mean_threshold(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_runtime_env(monkeypatch)
    _write_runtime_files(
        tmp_path,
        config_content=_replace_config_line(
            _CONFIG_CONTENT,
            old='sampled_phash_mean_threshold = 1.5',
            new='sampled_phash_mean_threshold = 63',
        ),
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(settings_module, 'CONFIG_PATH', tmp_path / 'config.toml')

    settings = Settings.load(is_dev=False)

    assert settings.sampled_phash_mean_threshold == 63


def test_settings_load_rejects_sampled_phash_mean_threshold_above_range(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_runtime_env(monkeypatch)
    _write_runtime_files(
        tmp_path,
        config_content=_replace_config_line(
            _CONFIG_CONTENT,
            old='sampled_phash_mean_threshold = 1.5',
            new='sampled_phash_mean_threshold = 64',
        ),
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(settings_module, 'CONFIG_PATH', tmp_path / 'config.toml')

    with pytest.raises(ValidationError):
        Settings.load(is_dev=False)
