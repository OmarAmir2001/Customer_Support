import cohere

from customer_support.helpers.logging_config import get_logger

from ..LLMEnum import CohereEnums, DocumentTypeEnum
from ..LLMInterface import LLMInterface


class CohereProvider(LLMInterface):
    def __init__(
        self,
        api_key: str,
        default_input_max_characters: int = 1000,
        default_generation_max_output_tokens: int = 1000,
        default_generation_temperature: float = 0.1,
    ):
        self.api_key = api_key
        self.default_input_max_characters = default_input_max_characters
        self.default_generation_max_output_tokens = default_generation_max_output_tokens
        self.default_generation_temperature = default_generation_temperature

        self.generation_model_id = None
        self.embedding_model_id = None
        self.embedding_size = None

        self.client = cohere.Client(api_key=self.api_key)

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
        chat_history: list | None = None,
        max_output_tokens: int = None,
        temperature: float = None,
    ):
        if not self.client:
            self.logger.error(
                "llm_client_not_initialised", provider="cohere", operation="generate_text"
            )
            return None
        if not self.generation_model_id:
            self.logger.error("generation_model_not_set", provider="cohere")
            return None

        max_output_tokens = (
            max_output_tokens if max_output_tokens else self.default_generation_max_output_tokens
        )
        temperature = temperature if temperature else self.default_generation_temperature

        # Normalised here rather than in the signature: a [] default is shared
        # across every call, so one mutation would leak into the next request.
        # Matches what OpenAIProvider already does.
        chat_history = chat_history or []

        response = self.client.chat(
            model=self.generation_model_id,
            chat_history=chat_history,
            message=self.construct_prompt(prompt, CohereEnums.USER.value),
            temperature=temperature,
            max_tokens=max_output_tokens,
        )
        if not response or not response.text:
            self.logger.error(
                "generation_response_invalid",
                provider="cohere",
                model=self.generation_model_id,
            )
            return None
        return response.text

    def generate_json(
        self,
        prompt: str,
        system_prompt: str = None,
        max_output_tokens: int = None,
        temperature: float = None,
    ) -> str:
        """Raw JSON string from the model, for the judges to validate.

        Cohere's v1 chat API has no ``response_format`` switch, so the JSON-only
        instruction has to travel in the preamble. The caller already fails closed on
        unparseable output, which is the backstop for that weaker guarantee.

        Like the OpenAI implementation, the prompt is NOT passed through
        ``process_text``: truncating a judge's evidence would invalidate its verdict.
        """
        if not self.client:
            self.logger.error(
                "llm_client_not_initialised", provider="cohere", operation="generate_json"
            )
            return None
        if not self.generation_model_id:
            self.logger.error("generation_model_not_set", provider="cohere")
            return None

        response = self.client.chat(
            model=self.generation_model_id,
            preamble=system_prompt,
            message=prompt,
            temperature=(
                temperature if temperature is not None else self.default_generation_temperature
            ),
            max_tokens=max_output_tokens or self.default_generation_max_output_tokens,
        )
        if not response or not response.text:
            self.logger.error(
                "judge_response_invalid",
                provider="cohere",
                model=self.generation_model_id,
            )
            return None
        return response.text

    def embed_text(self, text: str | list[str], document_type: str = None):
        if not self.client:
            self.logger.error(
                "llm_client_not_initialised", provider="cohere", operation="embed_text"
            )
            return None
        if isinstance(text, str):
            text = [text]
        if not self.embedding_model_id:
            self.logger.error("embedding_model_not_set", provider="cohere")
            return None
        if not self.embedding_size:
            self.logger.error(
                "embedding_size_not_set",
                provider="cohere",
                model=self.embedding_model_id,
            )
            return None

        input_type = CohereEnums.DOCUMENT.value
        if document_type == DocumentTypeEnum.QUERY.value:
            input_type = CohereEnums.QUERY.value

        texts = [text] if isinstance(text, str) else text
        response = self.client.embed(
            model=self.embedding_model_id,
            texts=[self.process_text(t) for t in texts],
            input_type=input_type,
            embedding_types=["float"],
        )
        if not response or not response.embeddings or not response.embeddings.float:
            self.logger.error(
                "embedding_response_invalid",
                provider="cohere",
                model=self.embedding_model_id,
            )
            return None
        return response.embeddings.float

    def construct_prompt(self, prompt: str, role: str):
        return {"role": role, "text": self.process_text(prompt)}
