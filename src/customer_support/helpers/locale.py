"""Language negotiation for incoming requests.

An HTTP concern, so it lives here and is used by the router — not by a controller.
The router resolves the locale once, writes it into the graph state, and everything
downstream reads that one value rather than re-deriving it.

The precedence chain below is the standard one:

1. an explicit per-request override — a client asking for one answer in one language
2. the student's stored preference (Section 6's ``preferred_language``)
3. the ``Accept-Language`` header the client sent
4. the configured primary language

The ordering rule that matters: a server must never silently override an explicit
choice. It also deliberately never infers language from IP or geography — a student
in Cairo may well want English, and country is not language.

Step 2 is wired as a parameter rather than read here, because profiles arrive with
MemoryController (Section 6). Until then the caller passes None and the chain simply
skips that rung.
"""

from customer_support.helpers.logging_config import get_logger

logger = get_logger(__name__)

BCP47_SEPARATOR = "-"


def parse_accept_language(header: str | None, max_tags: int = 10) -> list[str]:
    """Turn an Accept-Language header into language codes, best first.

    Handles the real header shape — ``ar-EG,ar;q=0.9,en;q=0.8`` — by sorting on the
    quality value and expanding each tag to its base language, so ``ar-EG`` can
    still match an ``ar`` locale (RFC 4647 lookup, simplified: this project's
    locales are plain language codes, so region subtags only ever widen a match).

    Malformed input yields an empty list rather than raising: a broken header from
    some client must not fail the request, it just means "no preference expressed".
    """
    if not header:
        return []

    weighted: list[tuple[float, int, str]] = []
    for position, part in enumerate(header.split(",")[:max_tags]):
        token = part.strip()
        if not token:
            continue

        tag, _, params = token.partition(";")
        tag = tag.strip()
        if not tag:
            continue

        quality = 1.0
        if params:
            key, _, raw = params.strip().partition("=")
            if key.strip().lower() == "q":
                try:
                    quality = float(raw)
                except ValueError:
                    quality = 1.0

        # q=0 means "explicitly not this language".
        if quality <= 0:
            continue

        # position keeps the original order stable among equal q values
        weighted.append((quality, -position, tag))

    ordered: list[str] = []
    for _, _, tag in sorted(weighted, reverse=True):
        if tag == "*":
            # "any language" adds no information beyond the configured default.
            continue
        for candidate in (tag, tag.split(BCP47_SEPARATOR, 1)[0]):
            normalised = candidate.strip().lower()
            if normalised and normalised not in ordered:
                ordered.append(normalised)

    return ordered


#: Where Arabic lives in Unicode: the main block, Arabic Supplement and Extended-A,
#: plus the Presentation Forms some sources use.
_ARABIC_RANGES = (("؀", "ۿ"), ("ݐ", "ݿ"), ("ࢠ", "ࣿ"), ("ﭐ", "﷿"), ("ﹰ", "﻿"))


def detect_question_language(text: str | None, threshold: float = 0.2) -> str | None:
    """Guess a language from the script the question is written in, or None.

    Only distinguishes Arabic from everything else, which is all this deployment's
    two locales require — and script detection is reliable in a way that
    word-frequency guessing is not.

    The threshold is deliberately low. A question is often mostly an English course
    code or a number with a few Arabic words around it ("كم ساعة في CS201؟"), and
    that is still an Arabic question.
    """
    if not text:
        return None

    letters = [c for c in text if c.isalpha()]
    if not letters:
        return None

    arabic = sum(1 for c in letters if any(lo <= c <= hi for lo, hi in _ARABIC_RANGES))
    return "ar" if arabic / len(letters) >= threshold else None


def negotiate_language(
    parser,
    requested: str | None = None,
    profile_language: str | None = None,
    accept_language: str | None = None,
    question: str | None = None,
) -> str:
    """Resolve the locale for one request, highest-priority signal first.

    ``parser`` is a TemplateParser — it owns which locales actually exist, so this
    function never has to know. Returns a language the parser supports.

    ``question`` is the lowest-priority signal, below every *stated* preference and
    above the configured default. It exists because of a real regression: a client
    that sends no language at all used to get Arabic answers to Arabic questions
    anyway, purely because the old generation model ignored the prompt's "Answer in
    English" instruction and matched the question instead. A model that obeys the
    instruction — command-r-plus does — exposed that the negotiation had never
    actually looked at the question. Answering an Arabic question in English is a
    worse default than guessing from its script.

    It stays last on purpose: an explicit request, a stored preference or an
    Accept-Language header is a person telling us what they want, and a detector must
    never override that. A bilingual student may well type Arabic and want English.
    """
    supported = parser.supported_languages

    candidates: list[str] = []
    if requested:
        candidates.append(requested.strip().lower())
    if profile_language:
        candidates.append(profile_language.strip().lower())
    candidates.extend(parse_accept_language(accept_language))

    # Tracked separately so the "nothing you asked for exists" warning below stays
    # about what the CLIENT asked for. A detected language was never requested, so
    # logging it as unavailable would be misleading.
    detected = detect_question_language(question)
    stated_candidates = list(candidates)
    if detected:
        candidates.append(detected)

    for candidate in candidates:
        if candidate in supported:
            return candidate

    # Nothing the client asked for is available; the parser applies primary/default.
    #
    # Logged HERE rather than left to the parser: this function is the only place
    # that knows what was actually asked for. Handing the parser requested=None
    # would resolve correctly and tell nobody, which is the silent half-translated
    # deployment this logging exists to prevent.
    resolved = parser.resolve_language(None)
    if stated_candidates:
        logger.warning(
            "locale_not_available",
            requested=requested,
            profile_language=profile_language,
            accept_language=accept_language,
            detected=detected,
            resolved=resolved,
            available=list(supported),
        )
    return resolved
