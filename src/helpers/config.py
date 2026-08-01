from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):

    APP_NAME: str
    APP_VERSION: str
    GROQ_API_KEY: str
    File_Allowed_Types: list[str]
    File_Max_Size: int
    FILE_Default_CHUNK_SIZE: int
    MONGODB_URL: str
    MONGODB_DATABASE: str

    GENERATION_BACKEND: str = None
    EMBEDDING_BACKEND: str = None

    OPENAI_API_URL: str = None
    GROQ_API_KEY: str = None  # this is openai api key for groq since i do not use oenai itself if you want to use openai dicrectly remeber to change lllmfactory
    COHERE_API_KEY: str = None

    GENERATION_MODEL_ID: str = None
    EMBEDDING_MODEL_ID: str = None
    EMBEDDING_MODEL_SIZE: int = None
    INPUT_DEFAULT_MAX_CHARACTERS: int = None
    GENERATION_DEFAULT_MAX_TOKENS: int = None
    GENERATION_DEFAULT_TEMPERATURE: float = None

    class Config:
        env_file = ".env"


def get_settings() -> Settings:
    return Settings()

