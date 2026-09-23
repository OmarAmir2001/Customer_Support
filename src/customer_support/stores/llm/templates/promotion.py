"""Promotion rubrics (Section 5).

Single-language and outside locales/, for the same reason the gate rubrics are:
their output is JSON read by an advisor dashboard, never prose shown to a student.

These two checks are NOT symmetrical, and the asymmetry is the whole design:

* **Generalizability only SUGGESTS.** It sets the default state of the advisor's
  "add to knowledge base" checkbox and nothing else. If it misjudges, the advisor
  flips the box and nothing is lost. If it could reject, a wrong call would silently
  discard good knowledge — and nobody would ever find out, because the failure looks
  like an answer that simply never got promoted.

* **Contradiction HOLDS.** It is the one exception to "just a default". A resolved
  answer that conflicts with the handbook is not a chunk to add alongside the
  handbook — it is a signal that one of the two is wrong, and a human needs to look
  at the *handbook itself*. Adding a competing chunk would leave retrieval free to
  surface either, which is exactly the drift Section 1 forbids.
"""

from string import Template

VERSION = "promotion/v1"

PROMOTION_SYSTEM_PROMPT = (
    "You review answers that a human academic advisor gave to a student, deciding "
    "whether each one belongs in a shared knowledge base. "
    "You reply with a single JSON object and nothing else: no prose, no markdown fences. "
    'The JSON has exactly two keys: "score" (a number between 0.0 and 1.0) and '
    '"reason" (one short sentence, max 40 words).'
)

# Sets the checkbox default. Never rejects.
GENERALIZABILITY_TEMPLATE = Template(
    "\n".join(
        [
            "Decide whether this advisor's answer is GENERAL knowledge that would help",
            "future students, or SPECIFIC to the one student who asked.",
            "",
            "Student's question:",
            "$question",
            "",
            "Advisor's answer:",
            "$answer",
            "",
            "Score 1.0 = clearly general, 0.0 = clearly specific to this student.",
            "",
            "Specific (score LOW) looks like:",
            "  - it names a person, or refers to 'your' particular circumstance",
            "  - it grants an exception, extension or accommodation to this student",
            "  - it depends on facts about this student (their grades, their situation)",
            "  - it is a one-off instruction: 'come to my office', 'send me your form'",
            "",
            "General (score HIGH) looks like:",
            "  - it states a rule, policy, deadline, procedure or definition",
            "  - it would be the same answer for any student asking the same thing",
            "",
            "Two traps:",
            "1. A general rule mentioned while answering a personal question is still",
            "   general. Judge the ANSWER's content, not the question's framing.",
            "2. A date or number does not make an answer specific. 'The deadline is",
            "   March 1st' is a general fact; 'your deadline is March 1st because of",
            "   your medical excuse' is not.",
            "",
            'Return JSON only: {"score": <float>, "reason": "<one sentence>"}',
        ]
    )
)

# Holds promotion and flags the handbook for review. The one check that can block.
CONTRADICTION_TEMPLATE = Template(
    "\n".join(
        [
            "Decide whether the advisor's answer CONTRADICTS the handbook excerpts.",
            "",
            "Handbook excerpts on the same topic:",
            "$chunks",
            "",
            "Advisor's answer:",
            "$answer",
            "",
            "Score 1.0 = the answer directly contradicts an excerpt, 0.0 = no conflict.",
            "",
            "Instructions:",
            "1. A contradiction means the two cannot both be true: a different number,",
            "   a different deadline, an opposite rule, a requirement one states and",
            "   the other denies.",
            "2. ADDING information the excerpts do not mention is NOT a contradiction.",
            "   Filling a gap is the entire point of promoting an answer — score it 0.0.",
            "3. Saying the same thing in different words is NOT a contradiction.",
            "4. Being more specific than the handbook is NOT a contradiction unless the",
            "   specifics conflict with what the handbook actually states.",
            "5. If the excerpts do not cover this topic at all, score 0.0.",
            "6. When you do find a conflict, the reason must name BOTH sides — what the",
            "   answer says and what the excerpt says — because a human will use it to",
            "   decide which of the two needs correcting.",
            "",
            'Return JSON only: {"score": <float>, "reason": "<one sentence>"}',
        ]
    )
)
