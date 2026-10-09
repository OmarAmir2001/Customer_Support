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

    # --- offline evaluation ---
    # Which model RAGAS uses to judge.
    #
    # Deliberately left on GROQ while the app itself runs on Cohere. That is not an
    # oversight: evaluation load and serving load then land on different providers,
    # so scoring a run cannot throttle the system being scored. Groq's free tier is
    # ample for this because nothing else uses it any more.
    #
    # Point it at a Cohere model if you want a single provider, but note RAGAS reaches
    # the model through LangChain rather than through this project's providers, so it
    # would need `langchain-cohere` installed.
    # Which provider RAGAS judges with. Separate from GENERATION_BACKEND because
    # evaluation and serving have different constraints.
    #
    # COHERE is the default because Groq's free tier cannot carry a sweep: RAGAS
    # prompts carry the full retrieved context, so four metrics over twenty questions
    # exceeded the 8,000 TPM cap within the second question. COHERE also avoids an
    # entire bug class — our Cohere provider is synchronous, so it has none of the
    # event-loop affinity that made the langchain_openai client fail between metrics.
    #
    # OPENAI (Groq) still works for small runs and keeps evaluation spend off the
    # paid key.
    RAGAS_BACKEND: str = "COHERE"
    RAGAS_MODEL_ID: str | None = None
    RAGAS_API_URL: str | None = None
    # RAGAS runs its metric jobs concurrently and defaults to 16 workers, which
    # produced a wall of TimeoutErrors and a `nan` for one whole metric — every job
    # for it failed.
    #
    # Note RAGAS does NOT go through this project's providers, so the retry policy in
    # stores/llm/rate_limit.py does not protect it. Concurrency control here is the
    # only lever, which is why this is low rather than the library default.
    RAGAS_MAX_WORKERS: int = 2
    RAGAS_TIMEOUT_SECONDS: int = 300
    # JSON metric verdicts need room after the reasoning tokens. These settings
    # apply only to the offline evaluator, not to the student-facing generator.
    RAGAS_MAX_OUTPUT_TOKENS: int = 4096
    RAGAS_REASONING_EFFORT: str | None = "low"

    # --- provider rate limiting ---
    # A load test found the reason these exist. On Groq's free tier the judge model
    # was capped at 8,000 tokens/minute, reached at FIVE concurrent users: of 94
    # questions only 5 were answered, and all 110 failures were 429s. The provider
    # replies "Please try again in 1.875s" — and the code ignored it, retrying twice
    # immediately so both retries hit the same limit. Every one of those became an
    # escalation, because the judges fail closed.
    #
    # The app now runs on a PAID Cohere key, so that specific ceiling is gone. These
    # settings stay, and still matter: a paid key has limits too, a separate trial-key
    # 429 on Cohere embeddings once killed an evaluation sweep mid-run, and a retry
    # policy is worth having before you need it rather than after.
    PROVIDER_RATE_LIMIT_MAX_RETRIES: int = 3
    # Per-sleep ceiling. The provider's hint is trusted only up to this: a bad or
    # hostile Retry-After must not be able to park a student's request for a minute.
    PROVIDER_RATE_LIMIT_MAX_WAIT_SECONDS: float = 8.0
    # Total budget across all retries for ONE call. Three gates run per question, so
    # an unbounded per-call wait multiplies into a wait the student actually notices.
    PROVIDER_RATE_LIMIT_TOTAL_BUDGET_SECONDS: float = 12.0

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

    # --- tracing (Langfuse) ---
    # Both keys empty means tracing is OFF and every call in helpers/tracing.py is a
    # no-op. That is the default on purpose: the test suite must not need a tracing
    # backend, and the project must run for someone who has not stood Langfuse up.
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_HOST: str = "http://localhost:3002"

    # --- escalation ---
    ESCALATION_STUDENT_MESSAGE: str = (
        "I couldn't answer this from the handbook with confidence, "
        "so I've passed it to an academic advisor. You'll find their reply here."
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
