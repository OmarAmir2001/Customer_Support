"""Answer generation from retrieved chunks. Separate from grading so the judges can
be tested without a generator and vice versa."""

import asyncio

from customer_support.helpers.logging_config import get_logger
from customer_support.models.db_schemas import RetrievedDocument
from customer_support.stores.llm.templates import RAG_GROUP, TemplateParser, format_excerpts

from .BaseController import BaseController


class GenerationController(BaseController):
    def __init__(self, generation_client, templates: TemplateParser, settings=None):
        super().__init__(settings)
        self.generation_client = generation_client
        self.templates = templates
        self.logger = get_logger(__name__)

    async def generate_answer(
        self, question: str, chunks: list[RetrievedDocument], language: str | None = None
    ) -> str | None:
        if not chunks:
            # Should be unreachable: gate 1 escalates first. Loud, not silent.
            raise ValueError("generate_answer called with no chunks")

        # Three parts, in the order the model reads them: who it is and its limits,
        # then the evidence, then the question with the grounding rule repeated. The
        # footer goes last on purpose — an instruction at the very end of a long
        # prompt is followed more reliably than the same one in the system message.
        system_prompt = self.templates.get(
            RAG_GROUP,
            "system_prompt",
            # The assistant's name is configuration, not prompt text, so it is
            # substituted in rather than written into each locale.
            {"assistant_name": self.app_settings.ASSISTANT_NAME},
            language=language,
        )

        # Same rendering the judges use, so an excerpt carries the same provenance
        # label on both sides of the gate.
        documents = format_excerpts(chunks, self.templates, language=language)

        footer = self.templates.get(
            RAG_GROUP, "footer_prompt", {"question": question}, language=language
        )

        if not system_prompt or not footer:
            # A missing template is a deployment defect, not a bad question. Fail
            # loudly here rather than sending the model a half-built prompt.
            raise RuntimeError(
                f"rag templates missing for language {language!r}; "
                "cannot build the answer prompt"
            )

        prompt = f"{documents}\n\n{footer}"

        answer = await asyncio.to_thread(
            self.generation_client.generate_text,
            prompt,
            [self.generation_client.construct_prompt(system_prompt, "system")],
            self.app_settings.GENERATION_DEFAULT_MAX_TOKENS,
            self.app_settings.GENERATION_DEFAULT_TEMPERATURE,
        )

        if not answer or not answer.strip():
            self.logger.error(
                "generation_empty_answer", chunk_count=len(chunks), language=language
            )
            return None

        return answer.strip()
