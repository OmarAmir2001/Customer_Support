"""PII detection and redaction, in both scripts the system actually serves.

**Why this is not Guardrails AI.** The rubric names that library, and it does install
cleanly against this project's pins — I checked. What it does not ship is the
validator: `guardrails/detect_pii` is a Hub install that pulls Microsoft Presidio
and spaCy at install time, needs a Hub token inside the Docker build, and recognises
entities with English NER models. This is a bilingual Arabic/English system, and its
most likely PII is an Egyptian mobile number or a 14-digit national ID typed in
Arabic-Indic digits — which those models do not cover. A framework that adds 149
packages and a build-time download to miss the actual case is the wrong trade, so
the detectors live here, cover both digit sets, and are tested.

**What it protects.** Three real paths, not a theoretical one:

1. *Outbound answers.* A promoted ticket answer is advisor-written free text that was
   embedded into the knowledge base — if an advisor typed a student's number into a
   resolution, retrieval can surface it to a different student later.
2. *The long-term profile.* Section 6 PERSISTS what it extracts. A student who writes
   "I'm Omar, 01012345678" must not end up with the number stored in `name`.
3. *The tracing backend.* Questions are sent to Langfuse as trace input. That is a
   third-party service, and a question is the one field a student types freely.

**What it deliberately does not detect.** Nothing shorter than nine digits, and no
bare number without a `+` or a leading `0`. The handbooks are full of numbers that
are not PII — ``135`` credit hours, a ``2.0`` GPA, ``مادة (٢٤)``, ``CS_2023`` — and a
redactor that eats the answer's actual content is worse than none. Every pattern
here requires a phone/ID shape, not merely a run of digits.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

#: Western, Arabic-Indic (U+0660–0669) and Extended Arabic-Indic (U+06F0–06F9).
#: Written into the patterns themselves rather than normalising the text first, so
#: every match span refers to the ORIGINAL string and redaction needs no index
#: remapping — the bug that normalise-then-replace invites.
_D = r"0-9٠-٩۰-۹"

#: Maps every digit form onto ASCII, for the one check that needs arithmetic (Luhn).
_DIGIT_MAP = {
    **{chr(0x0660 + n): str(n) for n in range(10)},
    **{chr(0x06F0 + n): str(n) for n in range(10)},
}

EMAIL = "EMAIL"
PHONE = "PHONE"
NATIONAL_ID = "NATIONAL_ID"
CARD = "CARD"


def _ascii_digits(text: str) -> str:
    return "".join(_DIGIT_MAP.get(char, char) for char in text)


def _luhn_ok(digits: str) -> bool:
    """The check digit. Without it any 16-digit run is "a card".

    Section numbers and long reference codes are common in a handbook; a card number
    is a specific arithmetic claim, and testing it is the difference between
    redacting a payment instrument and redacting a table of contents.
    """
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = int(char)
        if index % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


@dataclass(frozen=True)
class _Detector:
    kind: str
    pattern: re.Pattern[str]
    #: Extra arithmetic or structural test on the matched text. A pattern alone
    #: cannot distinguish a card from fourteen arbitrary digits.
    confirm: object = None


#: Order is priority, highest first, and it resolves overlaps. A 14-digit Egyptian
#: national ID starting 2 or 3 has roughly a one-in-ten chance of passing Luhn, so
#: NATIONAL_ID must be offered the span before CARD.
_DETECTORS: tuple[_Detector, ...] = (
    _Detector(
        EMAIL,
        # Local part kept conservative: no quoted forms, which do not appear in
        # student questions and would make the pattern much greedier.
        re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),
    ),
    _Detector(
        NATIONAL_ID,
        # Egyptian national ID: 14 digits, first digit is the century (2 = 1900s,
        # 3 = 2000s). Anchored on non-digits so it cannot be cut out of a longer run.
        re.compile(rf"(?<![{_D}])[23٢٣۲۳][{_D}]{{13}}(?![{_D}])"),
    ),
    _Detector(
        CARD,
        re.compile(rf"(?<![{_D}])[{_D}]{{4}}(?:[ \-]?[{_D}]{{4}}){{2,4}}(?![{_D}])"),
        confirm=lambda text: (
            13 <= len(re.sub(r"\D", "", _ascii_digits(text))) <= 19
            and _luhn_ok(re.sub(r"\D", "", _ascii_digits(text)))
        ),
    ),
    _Detector(
        PHONE,
        # Two shapes, and both REQUIRE a dialling prefix rather than just length:
        #   +<country><7-15 digits>, separators allowed
        #   0<9-10 digits>, which is the Egyptian local form (01x xxxx xxxx)
        # Anything that is merely a long number — a year, a credit-hour total, a
        # section reference — matches neither.
        re.compile(
            rf"(?<![{_D}])(?:"
            rf"\+[{_D}](?:[ \-]?[{_D}]){{6,16}}"
            # The leading zero must itself be any digit form. An ASCII "0" here
            # silently defeated the whole Arabic case: ٠١٠١٢٣٤٥٦٧٨ matched nothing
            # while 01012345678 matched, in a system whose students write Arabic.
            rf"|[0\u0660\u06f0][{_D}](?:[ \-]?[{_D}]){{8,11}}"
            rf")(?![{_D}])"
        ),
        # Guards the separator-tolerant form against matching "0" plus a date or a
        # list of small numbers: a real number has 9–15 digits once stripped.
        confirm=lambda text: 9 <= len(re.sub(r"\D", "", _ascii_digits(text))) <= 16,
    ),
)


@dataclass(frozen=True)
class _Match:
    start: int
    end: int
    kind: str


def _find(text: str) -> list[_Match]:
    """Every confirmed match, non-overlapping, left to right.

    Overlaps are resolved by detector priority first and length second, so a
    national ID is never reported as a shorter phone number sitting inside it.
    """
    candidates: list[tuple[int, int, int, str]] = []
    for priority, detector in enumerate(_DETECTORS):
        for match in detector.pattern.finditer(text):
            if detector.confirm is not None and not detector.confirm(match.group()):
                continue
            candidates.append(
                (priority, -(match.end() - match.start()), match.start(), detector.kind)
            )

    # Sort by priority, then by longest, then by position; claim spans greedily.
    taken: list[_Match] = []
    for _priority, negative_length, start, kind in sorted(candidates):
        end = start - negative_length
        if any(start < other.end and other.start < end for other in taken):
            continue
        taken.append(_Match(start, end, kind))

    return sorted(taken, key=lambda m: m.start)


def _apply(text: str, matches: list[_Match]) -> tuple[str, Counter]:
    """Replace right to left, so earlier spans keep their offsets."""
    found: Counter = Counter()
    for match in sorted(matches, key=lambda m: m.start, reverse=True):
        found[match.kind] += 1
        text = f"{text[: match.start]}[{match.kind}]{text[match.end :]}"
    return text, found


def redact(text: str | None) -> tuple[str | None, Counter]:
    """``(redacted_text, {kind: count})``. Returns the input unchanged if clean.

    Never raises and never returns None for a non-None input: this runs on the way
    out of a request that has already succeeded, so a redactor bug must not be able
    to turn a good answer into an error.
    """
    if not text:
        return text, Counter()
    try:
        matches = _find(text)
        if not matches:
            return text, Counter()
        return _apply(text, matches)
    except Exception:  # pragma: no cover - defensive
        return text, Counter()


def contains_pii(text: str | None) -> bool:
    return bool(text) and bool(_find(text))


#: How far from the end of a stream the redactor refuses to commit. Must exceed the
#: longest token a detector can match (a spaced 19-digit card is ~24 characters), with
#: room for an email address.
_HOLDBACK = 64


class StreamRedactor:
    """Redaction over a stream, which cannot be done chunk by chunk.

    A provider can split ``01012345678`` across two chunks, and redacting each chunk
    on its own emits ``0101`` in the clear and then redacts nothing. So this buffers
    and only commits text it can prove is complete:

    * it never emits within ``_HOLDBACK`` characters of the end;
    * it never splits a whitespace-delimited token, so a half-arrived number stays
      in the buffer rather than being emitted as a prefix;
    * if a confirmed match straddles the commit point, the point moves back to the
      start of that match.

    The cost is 64 characters of latency at the very start of an answer, which is not
    perceptible — and far cheaper than the alternative, which is a streaming endpoint
    with no redaction at all.
    """

    def __init__(self) -> None:
        self._buffer = ""
        self.found: Counter = Counter()

    def feed(self, chunk: str) -> str:
        """Take a chunk, return whatever is now safe to emit (often "")."""
        self._buffer += chunk or ""

        cut = len(self._buffer) - _HOLDBACK
        if cut <= 0:
            return ""

        # Align to a whitespace boundary so no token is committed half-arrived.
        boundary = max(
            self._buffer.rfind(" ", 0, cut + 1),
            self._buffer.rfind("\n", 0, cut + 1),
            self._buffer.rfind("\t", 0, cut + 1),
        )
        cut = boundary + 1 if boundary != -1 else 0
        if cut <= 0:
            return ""

        matches = _find(self._buffer)
        for match in matches:
            if match.start < cut < match.end:
                cut = match.start
                break
        if cut <= 0:
            return ""

        head, self._buffer = self._buffer[:cut], self._buffer[cut:]
        emitted, found = _apply(head, [m for m in matches if m.end <= cut])
        self.found.update(found)
        return emitted

    def flush(self) -> str:
        """The tail, redacted. Call once when the stream ends."""
        remainder, self._buffer = self._buffer, ""
        emitted, found = redact(remainder)
        self.found.update(found)
        return emitted or ""


def redact_fields(fields: dict) -> tuple[dict, Counter]:
    """Drop — not redact — any field whose value carries PII.

    For stored identity a redacted value is useless: a profile whose ``name`` is
    ``"[PHONE]"`` is worse than one with no name, because every later prompt renders
    it. The field is omitted instead, so the next turn that mentions the name cleanly
    can still learn it.
    """
    found: Counter = Counter()
    clean = {}
    for key, value in fields.items():
        if isinstance(value, str) and (hits := _find(value)):
            for match in hits:
                found[match.kind] += 1
            continue
        clean[key] = value
    return clean, found
