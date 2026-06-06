# schemas for request and response

from pydantic import BaseModel

class PromptRequest(BaseModel):
    question: str

class PromptResponse(BaseModel):
    response: str

class StatsResponse(BaseModel):
    chunk_size: int
    overlap_ratio: float
    top_k: int