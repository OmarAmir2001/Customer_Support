"""Judge prompts (Section 2), kept out of the controller so they can be versioned,
diffed, and later logged to MLflow as a parameter.

Each prompt asks for ONE thing, against explicit evidence, and returns JSON only.
The structure matters more than the wording: a judge that is asked to reason about
a concrete, checkable property is reliable where "rate your confidence" is not.
"""

def describe_source(doc) -> str:
    """Say what an excerpt IS, in words a judge or the generator can act on.

    Section 5 deliberately feeds two kinds of text into the same excerpt list:
    handbook sections and answers advisors gave to earlier students. A judge that
    cannot tell them apart treats the second kind as hearsay and marks a perfectly
    grounded answer unsupported — which kills the learning loop at gate 2a.

    Lives here rather than on either controller so the grader and the generator can
    label excerpts identically without importing each other.
    """
    meta = doc.metadata or {}
    source = meta.get("source") or "unknown source"

    if source == "instructor_resolved":
        return "answer previously given by an academic advisor"

    section = meta.get("section")
    label = f"student handbook {source}"
    return f"{label}, section {section}" if section else label


def format_excerpts(chunks) -> str:
    """Numbered, provenance-labelled excerpts. The numbering lets a judge point at a
    specific one; the label tells it what kind of source it is looking at."""
    return "\n\n".join(
        f"[{i}] ({describe_source(chunk)}) {chunk.text}"
        for i, chunk in enumerate(chunks, start=1)
    )


JUDGE_SYSTEM_PROMPT = (
    "You are a strict evaluator for a university handbook assistant. "
    "You reply with a single JSON object and nothing else: no prose, no markdown fences. "
    "The JSON has exactly two keys: "
    '"score" (a number between 0.0 and 1.0) and "reason" (one short sentence, max 40 words). '
    "Write the reason in the same language as the question."
)

# Gate 1 — runs BEFORE generation, so an unanswerable question never costs a
# generation call. Scores the retrieval, not the answer.
CONTEXT_RELEVANCE_PROMPT = """\
Decide whether the retrieved excerpts, TAKEN TOGETHER, contain enough material to
answer the student's question.

Each excerpt is labelled with what it is. Both kinds count equally here:
  - a section of the student handbook
  - an answer an academic advisor previously gave to this same question

Question:
{question}

Retrieved excerpts:
{chunks}

Instructions:
1. Judge COVERAGE, not precision: the question is what matters, not how many excerpts
   are on topic. One excerpt that fully answers it is a complete answer — the other
   excerpts being irrelevant does not make it less answerable.
2. Score on this scale:
   - 1.0  the excerpts fully answer the question
   - 0.7  the excerpts answer the main point, but a detail is missing
   - 0.4  the excerpts touch the topic but do not answer what was asked
   - 0.0  nothing here addresses the question at all
3. Do not reward or penalise the NUMBER of relevant excerpts.
4. If the score is below 0.4, the reason names what the handbook appears to be missing.

Return JSON only: {{"score": <float>, "reason": "<one sentence>"}}"""

# Gate 2a — claim decomposition. This decomposition is WHY faithfulness is reliable
# where a holistic "is this grounded?" score is not.
FAITHFULNESS_PROMPT = """\
Check whether every claim in the answer is supported by the excerpts.

Each excerpt is labelled with what it is. BOTH kinds are authoritative support:
  - a section of the student handbook
  - an answer an academic advisor previously gave (labelled as such)
An advisor's recorded answer is a valid source. Do NOT discount an excerpt for being
an advisor answer rather than handbook text, and do not require handbook wording.

Excerpts:
{chunks}

Answer to check:
{answer}

Instructions:
1. Break the answer into individual factual claims (ignore greetings and filler).
2. Mark each claim as SUPPORTED if ANY excerpt states or directly implies it.
   Your own general knowledge is not support. If every excerpt is silent on a claim,
   that claim is NOT supported.
3. score = (supported claims) / (total claims). If there are no factual claims, score is 1.0.
4. The reason must quote or name the FIRST unsupported claim, so an advisor can see what went wrong.

Return JSON only: {{"score": <float>, "reason": "<one sentence>"}}"""

# Gate 2b — deliberately does NOT see the excerpts. An answer can be perfectly
# grounded and still answer the wrong question; showing the excerpts here would
# reintroduce exactly the blind spot faithfulness already has.
ANSWER_RELEVANCE_PROMPT = """\
Decide whether the answer addresses the question that was actually asked.

Question:
{question}

Answer:
{answer}

Instructions:
1. Ignore whether the answer is true. Judge only whether it answers THIS question.
2. An answer that discusses the right topic but not the specific thing asked scores below 0.5.
   Example: asked "how many days do I have to withdraw?", answering "withdrawal is covered
   by the academic regulations" does not answer the question.
3. A refusal or a "contact the department" deflection scores 0.0.

Return JSON only: {{"score": <float>, "reason": "<one sentence>"}}"""

ANSWER_SYSTEM_PROMPT = (
    "You are a student support assistant for a computer science institute. "
    "Answer ONLY from the excerpts provided. Each excerpt is labelled with what it is: "
    "a section of the student handbook, or an answer an academic advisor previously gave. "
    "Both are authoritative — use whichever actually answers the question. "
    "If the excerpts do not contain the answer, say so plainly instead of guessing. "
    "Answer in the same language as the question. Be concise, and name the handbook "
    "section when the excerpt you used gives one."
)

ANSWER_PROMPT = """\
Excerpts:
{chunks}

Student question:
{question}

Write the answer using only the excerpts above."""