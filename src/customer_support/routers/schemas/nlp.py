from pydantic import BaseModel


class PushRequest(BaseModel):
    do_reset: int | None = 0


class SearchRequest(BaseModel):
    query: str
    limit: int | None = 10
    # Lets an admin exercise the department filter that real chat traffic goes through.
    department: str | None = None
