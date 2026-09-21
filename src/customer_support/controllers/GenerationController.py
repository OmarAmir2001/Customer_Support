"""Answer generation from retrieved chunks. Separate from grading so the judges can
be tested without a generator and vice versa."""

import asyncio

from customer_support.helpers.logging_config import get_logger
from customer_support.models.db_schemas import RetrievedDocument

from .BaseController import BaseController
from .judge_prompts import ANSWER_PROMPT, ANSWER_SYSTEM_PROMPT, format_excerpts


class GenerationController(BaseController):
    def __init__(self, generation_client, settings=None):
        super().__init__(settings)
        self.generation_client = generation_client
        self.logger = get_logger(__name__)

    async def generate_answer(self, question: str, chunks: list[RetrievedDocument]) -> str | None:
        if not chunks:
            # Should be unreachable: gate 1 escalates first. Loud, not silent.
            raise ValueError("generate_answer called with no chunks")

        # Same provenance labels the judges see, so the generator knows an advisor's
        # recorded answer is usable material and not something to hedge around.
        context = format_excerpts(chunks)
        prompt = ANSWER_PROMPT.format(question=question, chunks=context)

        answer = await asyncio.to_thread(
            self.generation_client.generate_text,
            prompt,
            [self.generation_client.construct_prompt(ANSWER_SYSTEM_PROMPT, "system")],
            self.app_settings.GENERATION_DEFAULT_MAX_TOKENS,
            self.app_settings.GENERATION_DEFAULT_TEMPERATURE,
        )

        if not answer or not answer.strip():
            self.logger.error("generation_empty_answer", chunk_count=len(chunks))
            return None

        return answer.strip()