# Security perspective

## F3 — `profile.department` is trusted, but memory extraction writes it

**Verdict: the precedence is correct and stays. The finding is real but mislocated —
the broken link is one level up, and it is the absence of an authenticated identity.**

### The chain as reported

```
request body is untrusted  →  stored profile overrides body  →  stored profile is
derived from the body's text by a small model
```

Confirmed in code. `PROFILE_FIELDS` includes `department`
(`models/llm_schemas/student_profile.py`), so `MemoryController.remember` can write
it from a student's own sentence, and `routers/chat.py` prefers it over
`payload.department`.

### Why the circularity is not the defect

Three things bound it, and all three hold today:

1. **Writing requires an assertion, not a hint.** Extraction is frequency-gated by a
   regex over first-person identity markers (`my`, `i study`, `تخصصي`, `قسمي`). A
   handbook question carries none, so the common path never reaches the model at all.
2. **The value is allowlisted, not sanitised.** `_known_department` normalises and
   then accepts only `CS` or `IS`; anything else becomes `None` rather than being
   stored. There is no string that reaches the retrieval filter except one of two.
3. **A null can never overwrite a known value.** `learned_fields()` strips nulls
   before the merge, so the destructive case — a wrong extraction erasing a right
   answer — is structurally impossible rather than merely unlikely.

So the worst a student can do *to their own profile* is tell the truth differently, or
lie and assert the other department.

### What that lie actually costs

**Confidentiality: nothing.** Both handbooks are documents the institute issues to
students. Reading the other one is not a disclosure.

**Integrity: the real cost, and it lands on the liar.** A CS student who gets
`department: IS` stored receives IS graduation requirements, IS credit-hour rules and
IS citations — confidently, and with the gates passing, because the answer is faithful
to the excerpts it was given. They are just the wrong excerpts. That is a correctness
failure dressed as a correct answer, which is the worst shape for one, and the system
has no way to notice: nothing downstream re-checks that the retrieved handbook matches
the student's real enrolment.

Mitigating it: `department_conflict` is logged whenever a stored value and a claimed
value disagree, so the discrepancy is visible after the fact.

### The actual defect

The report's chain starts at "request body is untrusted". It should start one step
earlier:

```
student_id is an UNAUTHENTICATED CLAIM  →  it selects whose profile is loaded,
and whose profile the background extraction writes
```

There is no authentication anywhere in this service — verified: no `Depends`-based
auth, no bearer scheme, no API key; the only `deny all` in nginx guards the metrics
path. `student_id` arrives in the request body and is used directly.

That turns F3 from a self-inflicted integrity risk into a **cross-student write
primitive**:

```
POST /api/v1/chat
{"student_id": "<someone else's id>", "question": "I study IS, when are exams?"}
```

The extraction runs against the victim's profile and stores `department: IS` on it.
Their next question is answered from the wrong handbook, and nothing in the system
attributes the change to the attacker — the profile carries no provenance for who
asserted a fact.

The same missing identity makes two reads trivial:

| | |
|---|---|
| `GET /api/v1/profile/{student_id}` | returns a named student's department and **GPA** |
| `GET /api/v1/chat/{thread_id}` | returns any student's full conversation |

And every advisor endpoint takes `advisor_id` as a *request parameter*, so any caller
can claim, resolve, reject or close any ticket as any advisor — and
`ticket_status_history` will faithfully record the impersonation as fact.

### Conclusion

**Do not "fix" F3 by distrusting the profile.** Preferring the body would reinstate
the hole the precedence was written to close — a student picking their handbook per
request — and would still leave the cross-student write, because the write is keyed on
an unauthenticated `student_id` either way. Distrusting both leaves department
unusable, and the department filter is the control that stops a CS student being
answered from the IS handbook at all.

The precedence is sound. What is missing is an authenticated subject to anchor it to.

### What closes it

In order. Only the first is load-bearing.

1. **Authenticate the student and derive `student_id` from the credential, never the
   body.** This single change removes the cross-student write, both reads above, and
   the advisor impersonation. Per the backend standard: *an ID in the URL or body is a
   claim about which resource the caller wants, not proof they may have it — check
   ownership against the authenticated user.*
2. **Authorize the advisor endpoints** by role, and take `advisor_id` from the token so
   the audit trail records a verified actor rather than a self-reported one.
3. **Record provenance on a learned field** — which turn asserted it, and when. The
   profile is the output of an inference and currently reads as fact.
4. *Optional.* Treat the registrar's enrolment record as authoritative for
   `department` once one is available, and demote both the profile and the body to
   fallbacks. This retires the circularity entirely, but it needs a system of record
   that does not exist yet.

Until (1) lands, this service belongs behind a network boundary that supplies the
identity — not on a student-facing route.

### Severity

| | |
|---|---|
| As reported (self-inflicted, authenticated world) | 🟢 low — bounded to two allowlisted values over public documents, with the conflict logged |
| As it stands (no authentication) | 🔴 **high** — cross-student profile write, GPA disclosure, conversation disclosure, advisor impersonation |

The severity is not a property of F3. It is a property of the missing authentication,
which F3 happens to expose a route into.

---

## SQL identifier interpolation in `PgVectorProvider`

Raised by dissection part 3. **Not currently exploitable; recorded so the day it
could be, this is the first place to look.**

### The shape

Nearly every statement in the file interpolates the table name directly:

```python
sql_text(f"SELECT COUNT(*) FROM {collection_name}")
sql_text(f"DELETE FROM {collection_name} WHERE ...")
```

This is unavoidable in kind: **table and column names cannot be bound as SQL
parameters.** Only values can. So any code that addresses a table chosen at runtime
must build an identifier into the statement text somehow.

### Why it is safe today

`collection_name` has exactly two sources, and neither is attacker-controlled:

| Source | Why it is trusted |
|---|---|
| `settings.KB_COLLECTION_NAME` | configuration, from `.env` at startup |
| `KBController.create_collection_name(project_id)` → `f"collection_{project_id}"` | `project_id` is a FastAPI `int` path parameter, so a non-integer is rejected with 422 before any handler body runs |

An `int` coerced into `f"collection_{...}"` cannot carry a quote, a semicolon or a
comment marker. The guarantee is **type coercion at the boundary**, not escaping.

### The inconsistency worth fixing eventually

`delete_collection` is the only method that quotes the identifier:

```python
safe_table_name = session.bind.dialect.identifier_preparer.quote(collection_name)
```

One method knows the right technique and thirteen do not use it. That asymmetry is
the actual risk: it reads as if quoting were optional, so the next person to add a
method copies a neighbour rather than the one correct example.

### What would make this exploitable

Any of these, and the file becomes the injection surface:

1. A collection name derived from a **request body or query string** rather than an
   `int` path parameter — a `tenant` or `collection` field, say, which is exactly
   what multi-tenancy would introduce (see `[[domain-agnostic-is-deferred]]`).
2. `project_id` widening from `int` to `str` to accommodate a slug.
3. A collection name read from the database, a config file someone else can write, or
   an admin UI field.

### Decision

Keep the interpolation; it is structurally necessary. Treat the **boundary type** as
the control, and make it explicit rather than incidental:

- Collection names are **trusted-by-source**, and the trust comes from `int`
  coercion at the HTTP boundary. That is a load-bearing property of
  `create_collection_name`, not an implementation detail.
- If a collection name ever becomes string-derived, route every statement in this
  file through `identifier_preparer.quote` first — and prefer an allowlist check
  against `list_all_collections()` over quoting alone.
- New methods should follow `delete_collection`, not its thirteen neighbours.
