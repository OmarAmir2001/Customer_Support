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

    # Documentation only — nothing reads these; they record which values the
    # setting above accepts. The default was `None`, which is not a valid list[str],
    # so each was effectively REQUIRED while declared optional. GENERATION_MODEL_ID_
    # LITERAL was also absent from .env.example, so a fresh clone could not boot the
    # app at all — and the error said "Input should be a valid list" rather than
    # "Field required", pointing at the type instead of the missing line.
    GENERATION_MODEL_ID_LITERAL: list[str] = []
    GENERATION_MODEL_ID: str | None = None
    EMBEDDING_MODEL_ID: str | None = None
    EMBEDDING_MODEL_SIZE: int | None = None
    INPUT_DEFAULT_MAX_CHARACTERS: int | None = None
    GENERATION_DEFAULT_MAX_TOKENS: int | None = None
    GENERATION_DEFAULT_TEMPERATURE: float | None = None

    # --- vector DB ---
    VECTOR_DB_BACKEND_LITERAL: list[str] = []
    VECTOR_DB_BACKEND: str
    VECTOR_DB_PATH: str
    VECTOR_DB_DISTANCE_METHOD: str | None = None
    VECTOR_DB_PGVEC_INDEX_THRESHOLD: int = 10000
    # Read by VectorDBProviderFactory for both backends. Must match
    # EMBEDDING_MODEL_SIZE, or inserts fail against the vector(N) column.
    VECTOR_DB_DEFAULT_VECTOR_SIZE: int = 384

    # --- logging ---
    LOG_LEVEL: str = "INFO"
    LOG_JSON: bool = True

    # --- long-term memory (Section 6) ---
    MEMORY_ENABLED: bool = True
    # Extraction is an easy task; answering handbook questions is not. Right-size the
    # model. Falls back to the judge model, then to the generation model.
    MEMORY_MODEL_ID: str | None = None
    MEMORY_MAX_OUTPUT_TOKENS: int = 600
    MEMORY_TEMPERATURE: float = 0.0

    # --- conversation history in the prompt ---
    # Two independent bounds. A turn cap alone lets six long advisor answers blow the
    # budget; a character cap alone spends it on fifty one-word turns. Both, and the
    # oldest turns are evicted until the rendered block fits.
    #
    # These bound the PROMPT. Storage is bounded separately by the messages reducer
    # (CONVERSATION_MAX_STORED_TURNS), which cannot read settings because it is
    # referenced in the state's type annotation at import time.
    CONVERSATION_HISTORY_TURNS: int = 6
    CONVERSATION_HISTORY_MAX_CHARS: int = 2000
    CONVERSATION_HISTORY_MAX_CHARS_PER_TURN: int = 400

    # --- assistant identity ---
    # Configuration, not prompt text: the name appears in both locales and in the
    # escalation message, and a name hardcoded in four places drifts. Substituted
    # into the prompts as $assistant_name.
    ASSISTANT_NAME: str = "Murshid"

    # --- language ---
    # PRIMARY_LANG is what a request with no stated preference gets. DEFAULT_LANG is
    # the floor the parser falls back to when a requested locale has no translation,
    # so it must be the one locale that is always complete.
    PRIMARY_LANG: str = "en"
    DEFAULT_LANG: str = "en"

    # --- retrieval ---
    KB_COLLECTION_NAME: str = "collection_1"
    RETRIEVAL_TOP_K: int = 5
    # Over-fetch before department filtering and source re-ranking discard rows.
    RETRIEVAL_OVERFETCH_FACTOR: int = 3

    # --- gates (Section 2) ---
    # Starting values. Tune them on the labelled eval set and log the winners to MLflow;
    # they are the single biggest lever on the escalation rate.
    GATE_CONTEXT_RELEVANCE_THRESHOLD: float = 0.5
    GATE_FAITHFULNESS_THRESHOLD: float = 0.8
    GATE_ANSWER_RELEVANCE_THRESHOLD: float = 0.7

    JUDGE_MODEL_ID: str | None = None  # falls back to GENERATION_MODEL_ID
    # Reasoning models spend tokens before emitting the JSON; too low a budget
    # truncates the object and the whole verdict is rejected.
    JUDGE_MAX_OUTPUT_TOKENS: int = 1200
    JUDGE_TEMPERATURE: float = 0.0  # a judge must be reproducible, not creative

    # --- promotion (Section 5) ---
    # Generalizability only moves a checkbox, so it can afford to be generous:
    # a wrongly-ticked box costs the advisor one glance, a wrongly-unticked one
    # silently loses knowledge.
    PROMOTION_GENERALIZABILITY_THRESHOLD: float = 0.6
    # Contradiction is a veto a human must clear, so it is deliberately harder to
    # trip than a gate threshold — a false hold blocks a real answer.
    PROMOTION_CONTRADICTION_THRESHOLD: float = 0.6
    # How much handbook context the contradiction check is shown.
    PROMOTION_CONTEXT_TOP_K: int = 5

    # --- escalation ---
    ESCALATION_STUDENT_MESSAGE: str = (
        "I couldn't answer this from the handbook with confidence, "
        "so I've passed it to an academic advisor. You'll find their reply here."
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
