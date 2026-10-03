from .LLMEnum import LLMEnums
from .providers import CohereProvider, OpenAIProvider


class LLMProviderFactory:
    def __init__(self, config: dict):
        self.config = config

    def create(self, provider_name: str):
        if provider_name == LLMEnums.OPENAI.value:
            return OpenAIProvider(
                api_key=self.config.GROQ_API_KEY,
                api_url=self.config.OPENAI_API_URL,
                default_input_max_characters=self.config.INPUT_DEFAULT_MAX_CHARACTERS,
                default_generation_max_output_tokens=self.config.GENERATION_DEFAULT_MAX_TOKENS,
                default_generation_temperature=self.config.GENERATION_DEFAULT_TEMPERATURE,
                # The 429 policy comes from Settings, so it is tunable per deployment
                # without a rebuild — the free tier and a paid tier want very
                # different numbers.
                rate_limit_max_retries=self.config.PROVIDER_RATE_LIMIT_MAX_RETRIES,
                rate_limit_max_wait=self.config.PROVIDER_RATE_LIMIT_MAX_WAIT_SECONDS,
                rate_limit_total_budget=self.config.PROVIDER_RATE_LIMIT_TOTAL_BUDGET_SECONDS,
            )
        if provider_name == LLMEnums.COHERE.value:
            return CohereProvider(
                api_key=self.config.COHERE_API_KEY,
                default_input_max_characters=self.config.INPUT_DEFAULT_MAX_CHARACTERS,
                default_generation_max_output_tokens=self.config.GENERATION_DEFAULT_MAX_TOKENS,
                default_generation_temperature=self.config.GENERATION_DEFAULT_TEMPERATURE,
            )
        return None
