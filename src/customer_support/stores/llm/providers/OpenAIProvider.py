from openai import OpenAI

from customer_support.helpers.logging_config import get_logger

from ..LLMEnum import OpenAIEnums
from ..LLMInterface import LLMInterface


class OpenAIProvider(LLMInterface):
    def __init__(
        self,
        api_key: str,
        api_url: str = None,
        default_input_max_characters: int = 1000,
        default_generation_max_output_tokens: int = 1000,
        default_generation_temperature: float = 0.1,
    ):
        self.api_key = api_key
        self.api_url = api_url

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

        response = self.client.chat.completions.create(
            model=self.generation_model_id,
            messages=messages,
            max_tokens=max_output_tokens,
            temperature=temperature,
        )
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

        response = self.client.chat.completions.create(
            model=self.generation_model_id,
            messages=messages,
            max_tokens=max_output_tokens or self.default_generation_max_output_tokens,
            temperature=(
                temperature if temperature is not None else self.default_generation_temperature
            ),
            response_format={"type": "json_object"},  # Groq supports this
        )
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
        response = self.client.embeddings.create(model=self.embedding_model_id, input=text)
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
