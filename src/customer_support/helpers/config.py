from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- app ---
    APP_NAME: str
    APP_VERSION: str

    # --- file upload ---
    FILE_ALLOWED_TYPES: list[str]
    FILE_MAX_SIZE: int
    FILE_DEFAULT_CHUNK_SIZE: int
    ASSETS_DIR: str = "assets"

    # --- postgres ---
    POSTGRES_USERNAME: str
    POSTGRES_PASSWORD: str
    POSTGRES_HOST: str
    POSTGRES_PORT: int
    POSTGRES_MAIN_DATABASE: str

    # --- LLM ---
    GENERATION_BACKEND: str | None = None
    EMBEDDING_BACKEND: str | None = None

    # Groq is reached through the OpenAI-compatible SDK, so this key is passed
    # to OpenAIProvider. If you switch to OpenAI itself, update LLMProviderFactory.
    GROQ_API_KEY: str
    OPENAI_API_URL: str | None = None
    COHERE_API_KEY: str | None = None

    GENERATION_MODEL_ID: str | None = None
    EMBEDDING_MODEL_ID: str | None = None
    EMBEDDING_MODEL_SIZE: int | None = None
    INPUT_DEFAULT_MAX_CHARACTERS: int | None = None
    GENERATION_DEFAULT_MAX_TOKENS: int | None = None
    GENERATION_DEFAULT_TEMPERATURE: float | None = None

    # --- vector DB ---
    VECTOR_DB_BACKEND: str
    VECTOR_DB_PATH: str
    VECTOR_DB_DISTANCE_METHOD: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()