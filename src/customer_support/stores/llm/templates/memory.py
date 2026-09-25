"""The memory extraction prompt (Section 6).

Single-language, and NOT under locales/, for the same reason the judge rubrics are
not: its output is a JSON object that no student ever reads. Only the answer is
student-facing. Translating this would double the surface to keep in step for zero
user-visible benefit.

The whole prompt is one discipline: **only record a fact that is clearly about this
student and clearly stated.** Trustcall's patching protects against a model dropping
a field it forgot to re-emit; it does nothing about a model confidently extracting
the wrong fact. "My friend is in IS" writing ``department: IS`` onto this student is
a memory bug no merge strategy can undo, so the defence has to live here, in the
prompt, and in the validators on StudentProfileUpdate.
"""

from string import Template

VERSION = "memory/v1"

MEMORY_SYSTEM_PROMPT = (
    "You extract durable identity facts about a student from a single message. "
    "You reply with one JSON object and nothing else: no prose, no markdown fences. "
    "You are conservative: reporting nothing is always better than reporting a guess."
)

EXTRACTION_TEMPLATE = Template(
    "\n".join(
        [
            "Read the student's message and report ONLY identity facts it states about",
            "the student themselves.",
            "",
            "Already known about this student (do not contradict without clear evidence):",
            "$known_profile",
            "",
            "Student's message:",
            "$message",
            "",
            "Fields you may report:",
            '  - "name"                the student\'s own name',
            '  - "department"          exactly "CS" or "IS"',
            '  - "gpa"                 a number between 0.0 and 4.0',
            '  - "preferred_language"  "en" or "ar"',
            "",
            "Rules:",
            "1. Report a field ONLY if the message states it about THIS student, clearly.",
            "   Omit the field otherwise. An omitted field means 'not mentioned' and",
            "   leaves what is already known untouched.",
            '2. A fact about someone else is NOT about this student. "My friend is in IS"',
            '   or "my brother\'s GPA is 3.9" report NOTHING.',
            "3. Do not infer. The language the message is written in is not a stated",
            "   preference. A course code is not a department. A mention of grades is",
            "   not a GPA.",
            "4. Do not record what the student ASKED about — only who they are. A question",
            "   about IS electives does not make the student an IS student.",
            "5. A pure question with no self-description reports nothing: {}",
            "",
            "Return JSON only, with just the fields you are confident about.",
            'Examples: {"name": "Omar", "department": "CS"}   or   {}',
        ]
    )
)
