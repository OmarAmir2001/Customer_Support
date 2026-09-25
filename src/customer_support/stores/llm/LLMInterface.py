from abc import ABC, abstractmethod


class LLMInterface(ABC):
    @abstractmethod
    def set_generation_model(self, model_id: str):
        pass

    @abstractmethod
    def set_embedding_model(self, model_id: str, embedding_size: int):
        pass

    @abstractmethod
    def generate_text(
        self,
        prompt: str,
        chat_history: list | None = None,
        max_output_tokens: int = None,
        temperature: float = None,
    ):
        pass

    @abstractmethod
    def generate_json(
        self,
        prompt: str,
        system_prompt: str = None,
        max_output_tokens: int = None,
        temperature: float = None,
    ) -> str:
        """Return the model's reply as a raw JSON string. The caller validates it.

        The judges need structured output, not prose — a judge whose verdict cannot be
        parsed is a judge that fails closed and escalates.
        """
        pass

    @abstractmethod
    def embed_text(self, text: str, document_type: str = None):
        pass

    @abstractmethod
    def construct_prompt(self, prompt: str, role: str):
        pass
