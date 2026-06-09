# schemas for request and response

from pydantic import BaseModel


class RootResponse(BaseModel):
    status: str
    service: str
    version: str
    supported_endpoints: list[str]


class PromptRequest(BaseModel):
    question: str


class AugmentedPrompt(BaseModel):
    System: str
    User: str


class DocumentResponse(BaseModel):
    article_id: str
    title: str
    chunk: str
    score: float


class PromptResponse(BaseModel):
    response: str
    context: list[DocumentResponse]
    Augmented_prompt: AugmentedPrompt


class StatsResponse(BaseModel):
    chunk_size: int
    overlap_ratio: float
    top_k: int
