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

    def stream_text(
        self,
        prompt: str,
        chat_history: list = None,
        max_output_tokens: int = None,
        temperature: float = None,
    ):
        """Yield the answer in pieces as the model produces it.

        NOT abstract, deliberately. Streaming is an optimisation of a user's
        perceived wait, not part of what makes a provider usable here — a provider
        that cannot stream should still be selectable. The default yields the whole
        answer as one chunk, so every caller can treat streaming as always available
        and no provider is forced to implement it.
        """
        answer = self.generate_text(
            prompt,
            chat_history=chat_history,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
        if answer:
            yield answer

    @abstractmethod
    def embed_text(self, text: str, document_type: str = None):
        pass

    @abstractmethod
    def construct_prompt(self, prompt: str, role: str):
        pass
