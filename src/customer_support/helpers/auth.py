"""Authentication for the advisor and operator routes.

**What this closes.** Every escalation and admin route was reachable by anyone who
could reach the port, and the actor was a *request field*: `advisor_id` in the body.
So a caller could claim a ticket, resolve it with `promote_to_kb: true`, and write
their own text into what future students are told — and `ticket_status_history` would
faithfully record whichever name they typed. "Human-gated promotion" was gated by
nothing, and the audit trail was a list of self-reported claims.

**The one rule this file exists to enforce**, from the backend standard: *an id in the
URL or body is a claim about which resource the caller wants, not proof they may have
it.* So the actor is never read from the body again — it comes from the credential,
and the credential is what identifies it.

**Deliberately an API key, not OAuth.** There is no identity provider here and no
user table for advisors. A key-to-id map in configuration is the smallest thing that
makes the audit trail truthful, and it is honest about what it is: a shared secret per
advisor, rotated by editing configuration. A real deployment puts an IdP in front and
replaces this module; the endpoints do not change, because they already take the
actor from a dependency rather than from the request.

**Fails closed.** No configured keys means no access, not open access. A missing
credential set reading as "authentication not required" is how most of these ship
open.
"""

from __future__ import annotations

from fastapi import Depends, Header, HTTPException, Request, status

from customer_support.helpers.logging_config import get_logger

logger = get_logger(__name__)

#: What the client sees on a failure. Deliberately identical for "no key" and "wrong
#: key": distinguishing them tells an attacker whether a key exists.
_UNAUTHORIZED = "A valid advisor API key is required for this endpoint."


async def advisor_identity(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> str:
    """Resolve the caller's API key to an advisor id, or raise 401.

    Returns the id to record as the actor. Every lifecycle endpoint takes this
    instead of a body field, so the recorded actor is the authenticated one by
    construction rather than by the endpoint remembering to check.
    """
    settings = getattr(request.app.state, "settings", None)
    if settings is None:  # pragma: no cover - the lifespan always sets this
        from customer_support.helpers.config import get_settings

        settings = get_settings()

    keys = getattr(settings, "ADVISOR_API_KEYS", None) or {}

    if not keys:
        # Fail closed, and say why — an operator seeing 401s on every advisor route
        # needs to know it is configuration rather than a bad key.
        logger.error(
            "advisor_auth_not_configured",
            detail=(
                "ADVISOR_API_KEYS is empty, so every advisor and admin route denies. "
                "Set it in .env; .env.example ships a development key."
            ),
            path=request.url.path,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=_UNAUTHORIZED,
            headers={"WWW-Authenticate": "ApiKey"},
        )

    if not x_api_key or x_api_key not in keys:
        logger.warning(
            "advisor_auth_rejected",
            path=request.url.path,
            # Never the key itself, and never a prefix of it — a prefix is enough to
            # confirm a guess.
            key_present=bool(x_api_key),
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=_UNAUTHORIZED,
            headers={"WWW-Authenticate": "ApiKey"},
        )

    advisor_id = keys[x_api_key]
    logger.info("advisor_authenticated", advisor_id=advisor_id, path=request.url.path)
    return advisor_id


#: Router-level dependency. Applied to the whole escalation and admin routers rather
#: than per endpoint, because "every route here needs auth" is a property of the
#: router, and a per-endpoint list is one someone forgets to add to.
RequireAdvisor = Depends(advisor_identity)
