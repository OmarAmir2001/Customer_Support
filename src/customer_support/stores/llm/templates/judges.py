"""Judge rubrics (Section 2).

Deliberately NOT under locales/. These are single-language on purpose: the only
part of a verdict a student ever sees is the ``reason`` string, and
``JUDGE_SYSTEM_PROMPT`` already asks for that in the question's language. Keeping
one copy of each rubric means tuning "score = supported / total" touches one place —
two translated copies would be two places to keep in step, and prompt drift is the
same failure as data drift.

``string.Template`` rather than ``str.format`` for a concrete reason: every rubric
ends by demanding literal JSON. Under ``.format`` each brace has to be doubled
(``{{"score": ...}}``), which silently breaks the first time someone edits a prompt
and writes a single brace. With ``$placeholder`` the JSON is written exactly as the
model should emit it.

Each prompt asks for ONE thing against explicit evidence. The structure matters more
than the wording: a judge reasoning about a concrete, checkable property is reliable
where "rate your confidence" is not.
"""

from string import Template

VERSION = "judges/v3"

JUDGE_SYSTEM_PROMPT = (
    "You are a strict evaluator for a university handbook assistant. "
    "You reply with a single JSON object and nothing else: no prose, no markdown fences. "
    'The JSON has exactly two keys: "score" (a number between 0.0 and 1.0) and '
    '"reason" (one short sentence, max 40 words). '
    "Write the reason in the same language as the question."
)

# Gate 1 — runs BEFORE generation, so an unanswerable question never costs a
# generation call. Scores the retrieval, not the answer.
CONTEXT_RELEVANCE_TEMPLATE = Template(
    "\n".join(
        [
            "Decide whether the retrieved excerpts, TAKEN TOGETHER, contain enough",
            "material to answer the student's question.",
            "",
            "Each excerpt is labelled with what it is. Both kinds count equally here:",
            "  - a section of the student handbook",
            "  - an answer an academic advisor previously gave to this same question",
            "",
            "Question:",
            "$question",
            "",
            "Retrieved excerpts:",
            "$chunks",
            "",
            "Instructions:",
            "1. Judge COVERAGE, not precision: what matters is the question, not how",
            "   many excerpts are on topic. One excerpt that fully answers it is a",
            "   complete answer — the others being irrelevant does not change that.",
            "2. Score on this scale:",
            "   - 1.0  the excerpts fully answer the question",
            "   - 0.7  the excerpts answer the main point, but a detail is missing",
            "   - 0.4  the excerpts touch the topic but do not answer what was asked",
            "   - 0.0  nothing here addresses the question at all",
            "3. Do not reward or penalise the NUMBER of relevant excerpts.",
            "4. If the score is below 0.4, the reason names what appears to be missing.",
            "5. EXCEPTION — if the question is about the ASSISTANT ITSELF (its name,",
            "   what it is, what it can do, or what it will not do), score 1.0. The",
            "   assistant answers those from its own description, so no excerpt is",
            "   needed and there is nothing for a human advisor to add.",
            "   This applies ONLY to questions about the assistant. A question about",
            "   institute policy that the excerpts do not cover still scores low.",
            "",
            'Return JSON only: {"score": <float>, "reason": "<one sentence>"}',
        ]
    )
)

# Gate 2a — claim decomposition. This decomposition is WHY faithfulness is reliable
# where a holistic "is this grounded?" score is not.
FAITHFULNESS_TEMPLATE = Template(
    "\n".join(
        [
            "Check whether every claim in the answer is supported by the excerpts.",
            "",
            "Each excerpt is labelled with what it is. BOTH kinds are authoritative:",
            "  - a section of the student handbook",
            "  - an answer an academic advisor previously gave (labelled as such)",
            "An advisor's recorded answer is a valid source. Do NOT discount an excerpt",
            "for being an advisor answer rather than handbook text, and do not require",
            "handbook wording.",
            "",
            "Excerpts:",
            "$chunks",
            "",
            "Answer to check:",
            "$answer",
            "",
            "Instructions:",
            "1. Break the answer into individual factual claims (ignore greetings and filler).",
            "2. Mark each claim as SUPPORTED if ANY excerpt states or directly implies it.",
            "   Your own general knowledge is not support. If every excerpt is silent on a",
            "   claim, that claim is NOT supported.",
            "3. score = (supported claims) / (total claims). If there are no factual",
            "   claims, score is 1.0.",
            "4. The reason must quote or name the FIRST unsupported claim, so an advisor",
            "   can see what went wrong.",
            "5. EXCEPTION — claims about the ASSISTANT ITSELF (its name, its role, what",
            "   it can do, what it refers to a human) are supported by definition: they",
            "   come from the assistant's own instructions, not from the excerpts. Do",
            '   NOT count them as unsupported. Without this, an answer to "who are',
            '   you?" would be scored as entirely ungrounded and escalated.',
            "",
            'Return JSON only: {"score": <float>, "reason": "<one sentence>"}',
        ]
    )
)

# Gate 2b — deliberately does NOT see the excerpts. An answer can be perfectly
# grounded and still answer the wrong question; showing the excerpts here would
# reintroduce exactly the blind spot faithfulness already has.
ANSWER_RELEVANCE_TEMPLATE = Template(
    "\n".join(
        [
            "Decide whether the answer addresses the question that was actually asked.",
            "",
            "Question:",
            "$question",
            "",
            "Answer:",
            "$answer",
            "",
            "Instructions:",
            "1. Ignore whether the answer is true. Judge only whether it answers THIS",
            "   question.",
            "2. An answer that discusses the right topic but not the specific thing asked",
            '   scores below 0.5. Example: asked "how many days do I have to withdraw?",',
            '   answering "withdrawal is covered by the academic regulations" does not',
            "   answer the question.",
            '3. A refusal or a "contact the department" deflection scores 0.0.',
            "",
            'Return JSON only: {"score": <float>, "reason": "<one sentence>"}',
        ]
    )
)
