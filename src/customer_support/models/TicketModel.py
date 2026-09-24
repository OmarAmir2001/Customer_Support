"""The only place that issues ticket SQL.

Controllers decide *whether* a ticket may change; this class performs the change.
It deliberately holds no lifecycle rules: putting them here would give the project
two places that know the state machine.
"""

from sqlalchemy import desc, select, update

from customer_support.helpers.logging_config import get_logger

from .BaseDataModel import BaseDataModel
from .db_schemas.customer_support.schemes.ticket import Ticket, TicketStatusHistory
from .enums.TicketStatusEnum import ConcurrentTicketUpdate, TicketStatus


class TicketModel(BaseDataModel):
    def __init__(self, db_client: object):
        super().__init__(db_client=db_client)
        self.logger = get_logger(__name__)

    @classmethod
    async def create_instance(cls, db_client: object):
        return cls(db_client=db_client)

    async def create_ticket(self, ticket: Ticket, actor: str) -> Ticket:
        """Insert the ticket and its first history row in ONE transaction.

        A ticket without an opening history row would be invisible to the stale
        scanner, which measures time since the last transition.
        """
        async with self.db_client() as session:
            async with session.begin():
                session.add(ticket)
                await session.flush()  # assigns ticket_id without ending the transaction

                session.add(
                    TicketStatusHistory(
                        ticket_id=ticket.ticket_id,
                        from_status=None,
                        to_status=ticket.status,
                        actor=actor,
                    )
                )
            await session.refresh(ticket)

        self.logger.info("ticket_created", ticket_id=ticket.ticket_id, thread_id=ticket.thread_id)
        return ticket

    async def get_ticket(self, ticket_id: int) -> Ticket | None:
        async with self.db_client() as session:
            result = await session.execute(select(Ticket).where(Ticket.ticket_id == ticket_id))
            return result.scalar_one_or_none()

    async def get_ticket_by_thread(self, thread_id: str) -> Ticket | None:
        async with self.db_client() as session:
            result = await session.execute(
                select(Ticket)
                .where(Ticket.thread_id == thread_id)
                .order_by(desc(Ticket.created_at))
                .limit(1)
            )
            return result.scalar_one_or_none()

    async def list_tickets(
        self,
        status: TicketStatus | None = None,
        department: str | None = None,
        promotion_held: bool | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> list[Ticket]:
        """Always paginated: an unbounded advisor dashboard is a slow query waiting
        to happen."""
        page = max(page, 1)
        page_size = min(max(page_size, 1), 100)

        stmt = select(Ticket)
        if status is not None:
            stmt = stmt.where(Ticket.status == status.value)
        if department is not None:
            stmt = stmt.where(Ticket.department == department)
        if promotion_held is not None:
            # The handbook review queue. `ix_ticket_promotion_held` is a PARTIAL index
            # on promotion_held IS TRUE, so the `is True` form is what can use it.
            stmt = stmt.where(
                Ticket.promotion_held.is_(True)
                if promotion_held
                else Ticket.promotion_held.is_(False)
            )

        stmt = stmt.order_by(Ticket.created_at).offset((page - 1) * page_size).limit(page_size)

        async with self.db_client() as session:
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def apply_transition(
        self,
        ticket_id: int,
        from_status: TicketStatus,
        to_status: TicketStatus,
        actor: str,
        note: str | None = None,
        fields: dict | None = None,
    ) -> Ticket:
        """Move a ticket, append history, and update fields — atomically.

        ``from_status`` is passed into the WHERE clause, so a concurrent update that
        already moved the ticket makes this a no-op instead of a lost update. Two
        advisors clicking Resolve at the same moment: the second one raises.
        """
        async with self.db_client() as session:
            async with session.begin():
                values = {"status": to_status.value}
                if fields:
                    values.update(fields)

                result = await session.execute(
                    update(Ticket)
                    .where(Ticket.ticket_id == ticket_id, Ticket.status == from_status.value)
                    .values(**values)
                    .returning(Ticket.ticket_id)
                )

                if result.scalar_one_or_none() is None:
                    # Typed, not RuntimeError: the router has to tell this apart from a
                    # genuine fault to answer 409 instead of 500.
                    raise ConcurrentTicketUpdate(ticket_id, from_status)

                session.add(
                    TicketStatusHistory(
                        ticket_id=ticket_id,
                        from_status=from_status.value,
                        to_status=to_status.value,
                        actor=actor,
                        note=note,
                    )
                )

            refreshed = await session.execute(select(Ticket).where(Ticket.ticket_id == ticket_id))
            ticket = refreshed.scalar_one()

        self.logger.info(
            "ticket_transition",
            ticket_id=ticket_id,
            from_status=from_status.value,
            to_status=to_status.value,
            actor=actor,
        )
        return ticket