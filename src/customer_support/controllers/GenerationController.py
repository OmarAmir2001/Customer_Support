"""Answer generation from retrieved chunks. Separate from grading so the judges can
be tested without a generator and vice versa."""

import asyncio

from customer_support.helpers.logging_config import get_logger
from customer_support.models.db_schemas import RetrievedDocument
from customer_support.models.graph.conversation import ConversationMessage
from customer_support.models.llm_schemas.student_profile import StudentProfile
from customer_support.stores.llm.templates import RAG_GROUP, TemplateParser, format_excerpts
from customer_support.stores.llm.templates.history import format_history

from .BaseController import BaseController


class GenerationController(BaseController):
    def __init__(self, generation_client, templates: TemplateParser, settings=None):
        super().__init__(settings)
        self.generation_client = generation_client
        self.templates = templates
        self.logger = get_logger(__name__)

    async def generate_answer(
        self,
        question: str,
        chunks: list[RetrievedDocument],
        language: str | None = None,
        profile: StudentProfile | None = None,
        messages: list[ConversationMessage] | None = None,
    ) -> str | None:
        if not chunks:
            # Should be unreachable: gate 1 escalates first. Loud, not silent.
            raise ValueError("generate_answer called with no chunks")

        system_prompt = self.templates.get(
            RAG_GROUP,
            "system_prompt",
            # The assistant's name is configuration, not prompt text, so it is
            # substituted in rather than written into each locale.
            {"assistant_name": self.app_settings.ASSISTANT_NAME},
            language=language,
        )

        footer = self.templates.get(
            RAG_GROUP, "footer_prompt", {"question": question}, language=language
        )

        if not system_prompt or not footer:
            # A missing template is a deployment defect, not a bad question. Fail
            # loudly here rather than sending the model a half-built prompt.
            raise RuntimeError(
                f"rag templates missing for language {language!r}; cannot build the answer prompt"
            )

        # Same rendering the judges use, so an excerpt carries the same provenance
        # label on both sides of the gate.
        documents = format_excerpts(chunks, self.templates, language=language)

        # Ordered so the model reads: who it is, who it is talking to, what was said
        # before, the evidence, and finally the question with the grounding rule.
        # The footer stays last on purpose — an instruction at the very end of a long
        # prompt is followed more reliably than the same one in the system message.
        sections = [
            self._profile_section(profile, language),
            self._history_section(messages, language),
            documents,
            footer,
        ]
        prompt = "\n\n".join(section for section in sections if section)

        answer = await asyncio.to_thread(
            self.generation_client.generate_text,
            prompt,
            [self.generation_client.construct_prompt(system_prompt, "system")],
            self.app_settings.GENERATION_DEFAULT_MAX_TOKENS,
            self.app_settings.GENERATION_DEFAULT_TEMPERATURE,
        )

        if not answer or not answer.strip():
            self.logger.error("generation_empty_answer", chunk_count=len(chunks), language=language)
            return None

        return answer.strip()

    def build_prompt(
        self,
        question: str,
        chunks: list[RetrievedDocument],
        language: str | None = None,
        profile: StudentProfile | None = None,
        messages: list[ConversationMessage] | None = None,
    ) -> tuple[str, str]:
        """The exact prompt generate_answer would send. Returns (system, prompt).

        Extracted so the streaming path cannot drift from the blocking one. Two
        copies of this assembly would diverge the first time a section was added,
        and the streamed answer would then be built from a different prompt than the
        one the judges were tuned against.
        """
        system_prompt = self.templates.get(
            RAG_GROUP,
            "system_prompt",
            {"assistant_name": self.app_settings.ASSISTANT_NAME},
            language=language,
        )
        footer = self.templates.get(
            RAG_GROUP, "footer_prompt", {"question": question}, language=language
        )
        if not system_prompt or not footer:
            raise RuntimeError(
                f"rag templates missing for language {language!r}; cannot build the answer prompt"
            )

        sections = [
            self._profile_section(profile, language),
            self._history_section(messages, language),
            format_excerpts(chunks, self.templates, language=language),
            footer,
        ]
        return system_prompt, "\n\n".join(section for section in sections if section)

    def stream_answer(
        self,
        question: str,
        chunks: list[RetrievedDocument],
        language: str | None = None,
        profile: StudentProfile | None = None,
        messages: list[ConversationMessage] | None = None,
    ):
        """Yield the answer in pieces, for a caller that streams to a client.

        **The judge gates do not run on this path, and that is the trade.** The gates
        are post-generation: faithfulness and answer relevance can only be scored
        once a complete answer exists. Streaming hands the student text before it
        exists, so there is nothing to withhold by the time a gate could object.

        So a streamed answer is an UNVERIFIED answer. `/chat` remains the gated,
        default path; streaming is opt-in for a caller that has accepted that trade —
        which is why it lives in the BentoML service rather than replacing /chat.
        """
        if not chunks:
            raise ValueError("stream_answer called with no chunks")

        system_prompt, prompt = self.build_prompt(
            question, chunks, language=language, profile=profile, messages=messages
        )
        yield from self.generation_client.stream_text(
            prompt,
            [self.generation_client.construct_prompt(system_prompt, "system")],
            self.app_settings.GENERATION_DEFAULT_MAX_TOKENS,
            self.app_settings.GENERATION_DEFAULT_TEMPERATURE,
        )

    # -------------------------------------------------------------- sections

    def _profile_section(self, profile: StudentProfile | None, language: str | None) -> str:
        """What we remember about this student, or "" when we remember nothing.

        Omitted entirely for an unknown student rather than rendered as an empty
        heading — and framed in the template as context, never as evidence, so the
        model cannot cite a remembered GPA as if the handbook stated it.
        """
        if profile is None or profile.is_empty:
            return ""

        described = profile.describe()
        if not described:
            return ""

        return (
            self.templates.get(
                RAG_GROUP, "profile_block", {"profile": described}, language=language
            )
            or ""
        )

    def _history_section(
        self, messages: list[ConversationMessage] | None, language: str | None
    ) -> str:
        """Recent turns, under a hard character budget.

        The CURRENT question is dropped: the router appended it to the transcript
        before the run, and it already appears in the footer. Including it twice
        invites the model to answer the previous turn instead.

        Budgeted rather than capped by turn count alone, because this competes for
        prompt space with the excerpts the answer is actually grounded in — and the
        provider truncates the whole prompt at INPUT_DEFAULT_MAX_CHARACTERS, so an
        unbounded history would silently push the evidence out.
        """
        if not messages or len(messages) < 2:
            return ""

        rendered = format_history(
            messages[:-1],
            self.templates,
            language=language,
            max_turns=self.app_settings.CONVERSATION_HISTORY_TURNS,
            max_chars=self.app_settings.CONVERSATION_HISTORY_MAX_CHARS,
            max_chars_per_turn=self.app_settings.CONVERSATION_HISTORY_MAX_CHARS_PER_TURN,
        )
        if not rendered:
            return ""

        return (
            self.templates.get(RAG_GROUP, "history_block", {"history": rendered}, language=language)
            or ""
        )
