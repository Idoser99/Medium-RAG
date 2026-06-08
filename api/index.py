# REST server
import os
from fastapi import FastAPI
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from api.schemas import PromptRequest, PromptResponse, StatsResponse, AugmentedPrompt, DocumentResponse
from api.engine import prompt_template, create_context
from pinecone import Pinecone
from scripts.embedder import Embedder
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

load_dotenv()  #loading .env file with configurations, base url, api key, remote paths etc.

app = FastAPI()

llm = ChatOpenAI(model="ZYRANGG-gpt-5-mini")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX_NAME")
index = pc.Index(name=index_name)
index_namespace = os.getenv("PINECONE_INDEX_NAMESPACE")
top_k = 14
vectorstore = PineconeVectorStore(
    index=index,
    embedding=OpenAIEmbeddings(model=Embedder.embedding_model),
    namespace=index_namespace,
    text_key="text"
)
high_threshold = 0.7
mid_threshold = 0.5
low_threshold = 0.35


@app.get("/ping")
def ping():
    return "pong"


@app.post("/api/prompt", response_model=PromptResponse)
def prompt(request: PromptRequest):
    prompt = request.question

    docs_scores_tuple = vectorstore.similarity_search_with_score(query=prompt, k=top_k)
    docs_scores_tuple = threshold_docs(docs_scores_tuple)
    docs = [doc for doc, score in docs_scores_tuple]
    context_segments = create_context(docs)
    context = "\n\n---\n\n".join(context_segments) if context_segments else "No relevant context found."
    augmented_prompt = prompt_template.format_messages(context=context, question=prompt)
    llm_response = llm.invoke(augmented_prompt)

    response = llm_response.content
    context_response = []
    for doc, score in docs_scores_tuple:
        context_response.append(
            DocumentResponse(
                article_id=str(doc.metadata.get("paper_id", "Unknown")),
                title=str(doc.metadata.get("title", "Unknown Title")),
                chunk=doc.page_content,
                score=float(score)
            )
        )
    system_prompt = augmented_prompt[0].content
    user_prompt = augmented_prompt[1].content
    return PromptResponse(
        response=response,
        context=context_response,
        Augmented_prompt=AugmentedPrompt(
            System=system_prompt,
            User=user_prompt
        )
    )


def threshold_docs(docs_scores_tuples: list[tuple[Document, float]]) -> list[tuple[Document, float]]:
    # high_docs = [(doc, score) for doc, score in docs_scores_tuples if score >= high_threshold]
    # if high_docs:
    #     return high_docs
    #
    # mid_docs = [(doc, score) for doc, score in docs_scores_tuples if score >= mid_threshold]
    # if mid_docs:
    #     return mid_docs

    low_docs = [(doc, score) for doc, score in docs_scores_tuples if score >= low_threshold]
    if low_docs:
        return low_docs

    return []


@app.get("/api/stats", response_model=StatsResponse)
def stats():
    response = StatsResponse(chunk_size=Embedder.chunk_size, overlap_ratio=Embedder.overlap_ratio, top_k=top_k)
    return response
