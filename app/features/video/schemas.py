from pydantic import BaseModel


class GenerateVideoBody(BaseModel):
    lesson_title: str
    lesson_content: str
    tenant_id: str | None = None
    max_slides: int = 6
