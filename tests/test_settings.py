from types import SimpleNamespace

import pytest
from pydantic import BaseModel, SecretStr, ValidationError

from timeline_hub.settings import S3Settings, Settings

_UNSET = object()


def _env_settings(
    *,
    bot_token: SecretStr | None | object = _UNSET,
) -> SimpleNamespace:
    if bot_token is _UNSET:
        bot_token = SecretStr('token')
    return SimpleNamespace(
        bot_token=bot_token,
        superuser_ids={1},
        user_ids={2},
        s3=S3Settings(
            endpoint_url='http://localhost:9000',
            region='us-east-1',
            bucket='bucket',
            access_key_id='key',
            secret_access_key=SecretStr('secret'),
        ),
    )


def test_settings_load_reads_bot_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('timeline_hub.settings._EnvSettings', lambda: _env_settings(bot_token=SecretStr('main-token')))

    settings = Settings.load()

    assert settings.bot_token.get_secret_value() == 'main-token'
    assert settings.superuser_ids == {1}
    assert settings.user_ids == {1, 2}


def test_settings_load_defaults_user_ids_and_folds_superusers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        'timeline_hub.settings._EnvSettings',
        lambda: SimpleNamespace(
            bot_token=SecretStr('main-token'),
            superuser_ids={7},
            user_ids=set(),
            s3=S3Settings(
                endpoint_url='http://localhost:9000',
                region='us-east-1',
                bucket='bucket',
                access_key_id='key',
                secret_access_key=SecretStr('secret'),
            ),
        ),
    )

    settings = Settings.load()

    assert settings.user_ids == {7}


def test_settings_load_propagates_validation_for_missing_required_env(monkeypatch: pytest.MonkeyPatch) -> None:
    class _MissingRequiredEnvSettings(BaseModel):
        bot_token: SecretStr

    monkeypatch.setattr('timeline_hub.settings._EnvSettings', _MissingRequiredEnvSettings)

    with pytest.raises(ValidationError):
        Settings.load()
