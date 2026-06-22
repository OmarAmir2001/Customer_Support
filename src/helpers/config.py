from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):

    APP_NAME: str
    APP_VERSION:str
    GROQ_API_KEY:str

    class config:
        env_file = ".env"

def get_settings() -> Settings:
    return Settings()

