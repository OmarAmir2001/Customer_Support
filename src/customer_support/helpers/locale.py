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


def negotiate_language(
    parser,
    requested: str | None = None,
    profile_language: str | None = None,
    accept_language: str | None = None,
) -> str:
    """Resolve the locale for one request, highest-priority signal first.

    ``parser`` is a TemplateParser — it owns which locales actually exist, so this
    function never has to know. Returns a language the parser supports.
    """
    supported = parser.supported_languages

    candidates: list[str] = []
    if requested:
        candidates.append(requested.strip().lower())
    if profile_language:
        candidates.append(profile_language.strip().lower())
    candidates.extend(parse_accept_language(accept_language))

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
    if candidates:
        logger.warning(
            "locale_not_available",
            requested=requested,
            profile_language=profile_language,
            accept_language=accept_language,
            resolved=resolved,
            available=list(supported),
        )
    return resolved
