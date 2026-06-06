from fastapi import FastAPI
from api.schemas import PromptRequest, PromptResponse, StatsResponse

app = FastAPI()

@app.get("/ping")
def ping():
    return "pong"

@app.post("/api/prompt",response_model=PromptResponse)
def prompt(request: PromptRequest):
    response = PromptResponse(response="your wife is right!")
    return response

@app.get("/api/stats", response_model=StatsResponse)
def stats():
    response = StatsResponse(chunk_size = 512, overlap_ratio=0.3, top_k=14)
    return response