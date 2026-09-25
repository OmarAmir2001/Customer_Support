from pydantic import BaseModel


class ProcessRequest(BaseModel):
    file_id: str = None
    chunk_size: int | None = 100
    overlap: int | None = 20
    do_reset: bool | None = False
