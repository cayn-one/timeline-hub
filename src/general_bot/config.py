from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    bot_token: str

    model_config = SettingsConfigDict(env_file='.env')


config = Config()  # type: ignore[call-arg]
