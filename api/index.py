# REST server

from fastapi import FastAPI
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from api.schemas import PromptRequest, PromptResponse, StatsResponse
from api.engine import prompt_template

load_dotenv() #loading .env file with configurations, base url, api key, remote paths etc.

app = FastAPI()

llm = ChatOpenAI(model="ZYRANGG-gpt-5-mini")

@app.get("/ping")
def ping():
    return "pong"

@app.post("/api/prompt",response_model=PromptResponse)
def prompt(request: PromptRequest):
    query = prompt_template.format_messages(question=request.question)
    ai_response = llm.invoke(query)
    response = PromptResponse(response=ai_response.content)
    return response

@app.get("/api/stats", response_model=StatsResponse)
def stats():
    response = StatsResponse(chunk_size = 512, overlap_ratio=0.3, top_k=14)
    return response