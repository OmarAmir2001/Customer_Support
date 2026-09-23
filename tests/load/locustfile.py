"""Load profiles for the handbook assistant.

    # read-only, free, safe to hammer
    uv run locust -f tests/load/locustfile.py --host http://127.0.0.1:8000 ReadOnlyUser

    # the full pipeline — SPENDS REAL MONEY and writes real tickets
    uv run locust -f tests/load/locustfile.py --host http://127.0.0.1:8000 ConversationUser

    # headless, e.g. 10 users for 2 minutes
    uv run locust -f tests/load/locustfile.py --host http://127.0.0.1:8000 \
        ReadOnlyUser --headless -u 10 -r 2 -t 2m

User classes are opt-in by name, and ``ReadOnlyUser`` is listed first on purpose:
running ``locust`` with no class named would otherwise start every class, and one of
them bills a model provider per request.

**The thing this file exists to measure.** The judges fail CLOSED: when one times
out or errors, the request escalates rather than failing. That is correct for a
student and invisible to an ordinary load test, which would report 100% success
while the system quietly stopped answering anything. So /chat is scored on its
BEHAVIOUR, not its status code — an escalation is recorded as a distinct,
non-successful outcome. A rising escalation rate under load is the real degradation
signal here, and the latency numbers alone will not show it.
"""

import logging
import random
import uuid

from locust import HttpUser, between, events, task

# Questions the indexed handbooks can answer. Mixed languages, because the prompt
# layer, the judges and the embedding model all behave differently per language and
# a single-language load test would miss half the cost.
ANSWERABLE = [
    "كم عدد الساعات المعتمدة المطلوبة للتخرج؟",
    "ما هي شروط الحصول على درجة البكالوريوس؟",
    "How many credit hours are required to graduate?",
    "What GPA do I need to stay in good standing?",
    "ما هي مدة التدريب الصيفي المطلوبة؟",
]

# Deliberately outside the handbooks. These SHOULD escalate, so a run with these
# mixed in exercises the ticket-writing path under load too.
UNANSWERABLE = [
    "How much does the student parking permit cost?",
    "What is the cafeteria menu today?",
]

DEPARTMENTS = ["CS", "IS", None]


@events.quitting.add_listener
def _assert_healthy(environment, **_kwargs):
    """Fail the run on a bad result, so headless/CI use has a real exit code.

    The zero-request case is checked FIRST, and it is not defensive padding. A
    1600-user ConversationUser run produced a completely empty stats table: every
    request was still in flight, and Locust only records a request when it
    FINISHES. Over zero requests ``fail_ratio`` is 0.0 and the percentile helper
    returns 0, so the worst outcome the system has — accepting load and completing
    none of it — was scoring as a clean pass and exiting 0. A stall has to be
    louder than a slow run, not silent.
    """
    stats = environment.stats.total

    if stats.num_requests == 0:
        logging.error(
            "no request completed: the server stalled, or every request is still "
            "in flight. This is a worse result than a high failure rate, not a better one."
        )
        environment.process_exit_code = 1
    elif stats.fail_ratio > 0.05:
        environment.process_exit_code = 1
    elif stats.get_response_time_percentile(0.95) > 30_000:
        # The pipeline makes up to five model calls; slow is expected, stalled is not.
        environment.process_exit_code = 1


class ReadOnlyUser(HttpUser):
    """Everything that costs nothing: no model calls, no writes.

    Safe to run at whatever concurrency you like. This is the profile that tells you
    about FastAPI, the connection pool and Postgres — the parts whose limits are not
    hidden behind a provider's rate limiter.
    """

    weight = 3
    wait_time = between(0.5, 2.0)

    def on_start(self):
        self.thread_ids: list[str] = []
        # Seed from tickets that already exist, so thread reads hit real checkpoints
        # instead of 404ing and flattering the latency numbers.
        with self.client.get(
            "/api/v1/escalation/tickets?page_size=20", name="/escalation/tickets [seed]",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                try:
                    self.thread_ids = [t["thread_id"] for t in response.json()][:20]
                except (ValueError, KeyError, TypeError):
                    pass
                response.success()
            else:
                response.failure(f"seed failed: {response.status_code}")

    @task(5)
    def health(self):
        self.client.get("/", name="/ [health]")

    @task(10)
    def list_tickets(self):
        page = random.randint(1, 3)
        self.client.get(
            f"/api/v1/escalation/tickets?ticket_status=pending&page={page}&page_size=20",
            name="/escalation/tickets",
        )

    @task(6)
    def read_thread(self):
        """Checkpointer read — deserialises the whole thread state."""
        if not self.thread_ids:
            return
        thread_id = random.choice(self.thread_ids)
        with self.client.get(
            f"/api/v1/chat/{thread_id}", name="/chat/{thread_id}", catch_response=True
        ) as response:
            # A thread that was trimmed away is a 404, not a defect.
            if response.status_code in (200, 404):
                response.success()
            else:
                response.failure(f"unexpected {response.status_code}")

    @task(4)
    def read_profile(self):
        self.client.get(
            f"/api/v1/profile/load-{random.randint(1, 50)}", name="/profile/{student_id}"
        )


class ConversationUser(HttpUser):
    """The full pipeline. COSTS REAL MONEY and creates real tickets.

    One request here is a query embedding, up to three judge calls and a generation
    call. Expect seconds, not milliseconds, and expect the provider's rate limiter to
    become the bottleneck well before the app does — which is itself worth knowing.
    """

    weight = 1
    # Long waits on purpose: this models a student thinking, not a scraper. Tightening
    # it mostly measures how fast the model provider will 429 you.
    wait_time = between(5, 15)

    def on_start(self):
        self.student_id = f"load-{uuid.uuid4().hex[:8]}"
        self.thread_id = str(uuid.uuid4())

    @task(8)
    def ask_answerable(self):
        self._ask(random.choice(ANSWERABLE), expect_answer=True)

    @task(2)
    def ask_unanswerable(self):
        """Exercises the escalation path: a ticket write plus a thread update."""
        self._ask(random.choice(UNANSWERABLE), expect_answer=False)

    @task(1)
    def follow_up(self):
        """Second turn on the same thread — loads history and re-reads the profile."""
        self._ask("And is that the same for part-time students?", expect_answer=True)

    def _ask(self, question: str, expect_answer: bool):
        payload = {
            "question": question,
            "student_id": self.student_id,
            "thread_id": self.thread_id,
            "department": random.choice(DEPARTMENTS),
        }
        label = "/chat [answerable]" if expect_answer else "/chat [escalates]"

        with self.client.post(
            "/api/v1/chat", json=payload, name=label, catch_response=True
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}")
                return

            try:
                body = response.json()
            except ValueError:
                response.failure("response was not JSON")
                return

            escalated = bool(body.get("escalated"))

            if expect_answer and escalated:
                # The signal this whole file exists for. Under load the judges start
                # timing out, fail closed, and every question turns into a ticket —
                # a total behavioural failure that returns 200 the entire time.
                response.failure("escalated a question the handbook covers")
            elif not expect_answer and not escalated:
                response.failure("answered a question the handbook does not cover")
            elif not body.get("answer"):
                response.failure("empty answer")
            else:
                response.success()
