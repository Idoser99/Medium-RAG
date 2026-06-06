# REST server
import os
from fastapi import FastAPI
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from api.schemas import PromptRequest, PromptResponse, StatsResponse
from api.engine import prompt_template
from pinecone import Pinecone
from scripts.embedder import Embedder
from langchain_pinecone import PineconeVectorStore

load_dotenv() #loading .env file with configurations, base url, api key, remote paths etc.

app = FastAPI()

llm = ChatOpenAI(model="ZYRANGG-gpt-5-mini")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX_NAME")
index_namespace = os.getenv("PINECONE_INDEX_NAMESPACE")
top_k = 14
vectorstore = PineconeVectorStore(
    index=index_name,
    embedding=Embedder.embeddings,
    namespace=index_namespace,
    text_key="text"
)

@app.get("/ping")
def ping():
    return "pong"

@app.post("/api/prompt",response_model=PromptResponse)
def prompt(request: PromptRequest):

    # prompt embedding
    # retrieve top k chunks
    # augmented prompt
    prompt = request.question
    docs = vectorstore.similarity_search(query=prompt, k=top_k)
    chunks = [doc.page_content for doc in docs]
    context = "\n\n---\n\n".join(chunks) if chunks else "No relevant context found."
    augmented_prompt = prompt_template.format_messages(context=context, question=prompt)
    llm_response = llm.invoke(augmented_prompt)
    response = PromptResponse(response=llm_response.content)
    return response

@app.get("/api/stats", response_model=StatsResponse)
def stats():
    response = StatsResponse(chunk_size = Embedder.chunk_size, overlap_ratio=Embedder.overlap_ratio, top_k=top_k)
    return response