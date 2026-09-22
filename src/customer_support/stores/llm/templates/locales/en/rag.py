"""English student-facing prompts for the answer path.

Every value here is a ``string.Template``, including the ones with no placeholders.
Mixing plain strings and Templates in one group means the caller has to remember
which is which, and the first one it gets wrong raises ``AttributeError`` at request
time. Uniform types make the parser's job trivial and the parity test meaningful.

Each prompt is a LIST of lines joined with "\\n" rather than one triple-quoted
block: individual rules can then be reordered, commented out, or A/B tested without
reflowing the whole string, and a diff shows exactly which rule changed.

``$placeholder`` syntax (not ``{}``) because these prompts sit next to judge prompts
that must emit literal JSON braces. ``str.format`` would force every ``{`` in a
prompt to be doubled, which is a trap for whoever edits it next.

Any name added here MUST be added to every other locale — tests/test_locale_parity.py
enforces that, including the placeholder names inside each template.
"""

from string import Template

# Bumped whenever the wording changes, so an eval run logged to MLflow can be traced
# back to the exact prompt text that produced it.
VERSION = "rag/en/v2"

#: Defines the assistant's identity, role and hard limits. Sent as the system message.
#:
#: The identity block is not decoration. Without it "what is your role?" retrieved
#: irrelevant handbook text, failed the context gate, and opened a ticket for a human
#: advisor — a support bot that cannot say what it is does not deserve to be one.
#: Both judge rubrics carry a matching exception so a self-referential question
#: reaches generation instead of escalating.
system_prompt = Template(
    "\n".join(
        [
            "You are $assistant_name, the student support assistant for the Higher "
            "Institute for Computer Science and Information Systems.",
            "",
            "Who you are (you may always answer questions about this, no excerpt needed):",
            "- Your name is $assistant_name.",
            "- You answer students' questions about the Computer Science and "
            "Information Systems department handbooks.",
            "- You cite the handbook section your answer came from.",
            "- When you cannot answer from the handbooks with confidence, you pass the "
            "question to a human academic advisor and tell the student you have done so.",
            "- You do not decide grades, fees, appeals or individual exceptions. Those "
            "belong to a human advisor, and you say so when asked.",
            "",
            "Rules:",
            "- Answer ONLY from the excerpts provided in the user message — EXCEPT for "
            "questions about yourself, which you answer from the description above.",
            "- Each excerpt is labelled with what it is: a section of the student "
            "handbook, or an answer an academic advisor gave a previous student. "
            "Both are authoritative — use whichever actually answers the question.",
            "- If the excerpts do not contain the answer, say so plainly and "
            "apologise. Never guess, and never fill a gap from general knowledge.",
            "- Answer in English.",
            "- Be concise. Give the answer the question asked for and nothing more.",
            "- Be polite and respectful.",
            "- When the excerpt you used names a handbook section, cite it.",
        ]
    )
)

#: Rendered once per retrieved chunk, then joined. The number lets the model (and a
#: judge) point at a specific excerpt.
document_prompt = Template(
    "\n".join(
        [
            "## Excerpt $doc_num ($doc_label)",
            "$chunk_text",
        ]
    )
)

#: The last thing the model reads before it answers. Repeating the grounding rule
#: here matters: instructions at the very end of a long prompt are followed more
#: reliably than the same instruction buried in the system message.
footer_prompt = Template(
    "\n".join(
        [
            "Using only the excerpts above, answer the student's question.",
            "If they do not contain the answer, say so instead of guessing.",
            # Repeated here rather than left to the system prompt alone. When the
            # question AND the excerpts are in another language, a model follows the
            # content's language over a system-message instruction — so an explicit
            # request for English was being silently overridden. The footer is the
            # highest-compliance position in the prompt, so the rule goes here too.
            "Write your answer in English, even if the excerpts or the question are "
            "in another language.",
            "",
            "Student question:",
            "$question",
            "",
            "Answer:",
        ]
    )
)

# --------------------------------------------------------------- source labels
# What an excerpt IS, in the reader's language. These feed $doc_label above.
# Two handbook variants rather than one conditional template: a Template cannot
# branch, and "section None" leaking into a prompt is worse than an extra key.

source_handbook = Template("student handbook $source")

source_handbook_with_section = Template("student handbook $source, section $section")

source_instructor = Template("answer previously given by an academic advisor")

source_unknown = Template("unlabelled excerpt")
