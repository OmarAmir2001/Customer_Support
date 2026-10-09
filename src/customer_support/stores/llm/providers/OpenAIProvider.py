from openai import OpenAI

from customer_support.helpers.logging_config import get_logger
from customer_support.utils.metrics import record_tokens

from ..LLMEnum import OpenAIEnums
from ..LLMInterface import LLMInterface
from ..rate_limit import call_with_rate_limit_retry


class OpenAIProvider(LLMInterface):
    def __init__(
        self,
        api_key: str,
        api_url: str = None,
        default_input_max_characters: int = 1000,
        default_generation_max_output_tokens: int = 1000,
        default_generation_temperature: float = 0.1,
        rate_limit_max_retries: int = 3,
        rate_limit_max_wait: float = 8.0,
        rate_limit_total_budget: float = 12.0,
    ):
        self.api_key = api_key
        self.api_url = api_url

        # Retry policy for 429s, applied to every call this provider makes. Defaults
        # match Settings so a provider constructed directly (in a test, or a script)
        # behaves like the running app rather than with no policy at all.
        self.rate_limit_max_retries = rate_limit_max_retries
        self.rate_limit_max_wait = rate_limit_max_wait
        self.rate_limit_total_budget = rate_limit_total_budget

        self.default_input_max_characters = default_input_max_characters
        self.default_generation_max_output_tokens = default_generation_max_output_tokens

        self.default_generation_temperature = default_generation_temperature

        self.generation_model_id = None

        self.embedding_model_id = None
        self.embedding_size = None

        self.client = OpenAI(api_key=self.api_key, base_url=self.api_url)
        self.logger = get_logger(__name__)

    def set_generation_model(self, model_id: str):
        self.generation_model_id = model_id

    def set_embedding_model(self, model_id: str, embedding_size: int):
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size

    def process_text(self, text: str):
        return text[: self.default_input_max_characters].strip()

    def _call(self, call, operation: str):
        """Every SDK call goes through here, so no call site can forget the policy.

        Only rate limits are retried; everything else propagates untouched, which
        leaves each caller's own error handling — including the judges' fail-closed
        path — behaving exactly as before.
        """
        return call_with_rate_limit_retry(
            call,
            operation=operation,
            max_retries=self.rate_limit_max_retries,
            max_wait=self.rate_limit_max_wait,
            total_budget=self.rate_limit_total_budget,
        )

    def _record_usage(self, response, model: str | None) -> None:
        """Report what the call cost, in tokens.

        Reads the provider's own numbers rather than estimating from characters: an
        estimate is wrong by a factor that varies with language, and Arabic
        tokenises very differently from English in these models — the one place a
        character-based guess would be most misleading here.

        Wrapped in its own try/except because this is instrumentation on a path that
        has already succeeded. A missing or renamed usage field must not turn a good
        answer into an error.
        """
        try:
            usage = getattr(response, "usage", None)
            if usage is None:
                return
            record_tokens(
                model,
                getattr(usage, "prompt_tokens", None),
                getattr(usage, "completion_tokens", None),
            )
        except Exception:  # pragma: no cover - instrumentation must never raise
            pass

    def generate_text(
        self,
        prompt: str,
        chat_history: list = None,
        max_output_tokens: int = None,
        temperature: float = None,
    ):
        if not self.client:
            self.logger.error(
                "llm_client_not_initialised", provider="openai", operation="generate_text"
            )
            return None
        if not self.generation_model_id:
            self.logger.error("generation_model_not_set", provider="openai")
            return None
        max_output_tokens = (
            max_output_tokens if max_output_tokens else self.default_generation_max_output_tokens
        )
        temperature = temperature if temperature else self.default_generation_temperature

        chat_history = chat_history or []
        messages = chat_history + [self.construct_prompt(prompt, OpenAIEnums.USER.value)]

        response = self._call(
            lambda: self.client.chat.completions.create(
                model=self.generation_model_id,
                messages=messages,
                max_tokens=max_output_tokens,
                temperature=temperature,
            ),
            operation="generate_text",
        )
        self._record_usage(response, self.generation_model_id)
        if (
            not response
            or not response.choices
            or len(response.choices) == 0
            or not response.choices[0].message
        ):
            self.logger.error(
                "generation_response_invalid",
                provider="openai",
                model=self.generation_model_id,
            )
            return None
        return response.choices[0].message.content

    def stream_text(
        self,
        prompt: str,
        chat_history: list = None,
        max_output_tokens: int = None,
        temperature: float = None,
    ):
        """Real token streaming from the provider.

        Note what is NOT wrapped in self._call: a streaming response cannot be
        retried once it has started yielding, because the caller has already been
        handed part of the answer. The create() call itself is the only retryable
        moment, so only that is wrapped — anything failing mid-stream propagates, and
        the caller decides whether a partial answer is usable.
        """
        if not self.client or not self.generation_model_id:
            self.logger.error(
                "llm_client_not_initialised", provider="openai", operation="stream_text"
            )
            return

        chat_history = chat_history or []
        messages = chat_history + [self.construct_prompt(prompt, OpenAIEnums.USER.value)]

        stream = self._call(
            lambda: self.client.chat.completions.create(
                model=self.generation_model_id,
                messages=messages,
                max_tokens=max_output_tokens or self.default_generation_max_output_tokens,
                temperature=(
                    temperature if temperature is not None else self.default_generation_temperature
                ),
                stream=True,
            ),
            operation="stream_text",
        )

        # NOT counted in customer_support_tokens_total. A streamed response carries
        # no usage unless `stream_options={"include_usage": True}` is requested, and
        # a provider that rejects an unknown parameter would fail the stream rather
        # than the counting — the wrong thing to risk for a cost panel. /chat is the
        # path that carries real traffic and it is counted.
        for chunk in stream:
            # Reasoning models emit chunks with no choices, and the final chunk
            # carries only a finish_reason. Both are normal, not errors.
            if not chunk.choices:
                continue
            piece = chunk.choices[0].delta.content
            if piece:
                yield piece

    def generate_json(
        self,
        prompt: str,
        system_prompt: str = None,
        max_output_tokens: int = None,
        temperature: float = None,
    ) -> str:
        """Raw JSON string from the model, for the judges to validate.

        Note this does NOT route the prompt through ``process_text``: that truncates
        to INPUT_DEFAULT_MAX_CHARACTERS (1024), and a judge prompt carrying five
        handbook excerpts is far longer than that. Silently cutting a judge's
        evidence in half would make its verdict meaningless.
        """
        if not self.client:
            self.logger.error(
                "llm_client_not_initialised", provider="openai", operation="generate_json"
            )
            return None
        if not self.generation_model_id:
            self.logger.error("generation_model_not_set", provider="openai")
            return None

        messages = []
        if system_prompt:
            messages.append({"role": OpenAIEnums.SYSTEM.value, "content": system_prompt})
        messages.append({"role": OpenAIEnums.USER.value, "content": prompt})

        response = self._call(
            lambda: self.client.chat.completions.create(
                model=self.generation_model_id,
                messages=messages,
                max_tokens=max_output_tokens or self.default_generation_max_output_tokens,
                temperature=(
                    temperature if temperature is not None else self.default_generation_temperature
                ),
                response_format={"type": "json_object"},  # Groq supports this
            ),
            # The judge path — the one the load test showed collapsing into
            # escalations under a 429.
            operation="generate_json",
        )
        self._record_usage(response, self.generation_model_id)
        if not response or not response.choices or not response.choices[0].message:
            self.logger.error(
                "judge_response_invalid",
                provider="openai",
                model=self.generation_model_id,
            )
            return None
        return response.choices[0].message.content

    def embed_text(self, text: str | list[str], document_type: str = None):
        if not self.client:
            self.logger.error(
                "llm_client_not_initialised", provider="openai", operation="embed_text"
            )
            return None

        if isinstance(text, str):
            text = [text]

        if not self.embedding_model_id:
            self.logger.error("embedding_model_not_set", provider="openai")
            return None
        if not self.embedding_size:
            self.logger.error(
                "embedding_size_not_set",
                provider="openai",
                model=self.embedding_model_id,
            )
            return None
        response = self._call(
            lambda: self.client.embeddings.create(model=self.embedding_model_id, input=text),
            operation="embed_text",
        )
        self._record_usage(response, self.embedding_model_id)
        if (
            not response
            or not response.data
            or len(response.data) == 0
            or not response.data[0].embedding
        ):
            self.logger.error(
                "embedding_response_invalid",
                provider="openai",
                model=self.embedding_model_id,
            )
            return None

        return [rec.embedding for rec in response.data]

    def construct_prompt(self, prompt: str, role: str):
        return {"role": role, "content": self.process_text(prompt)}
