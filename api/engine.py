# utils file for llm

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# Define your strict system prompt with a placeholder for the RAG context
SYSTEM_PROMPT = """
You are a Medium-article assistant that answers questions strictly and only
based on the Medium articles dataset context provided to you (metadata
and article passages). You must not use any external knowledge, the open
internet, or information that is not explicitly contained in the retrieved
context. If the answer cannot be determined from the provided context,
respond: “I don’t know based on the provided Medium articles data.”
Always explain your answer using the given context, quoting or
paraphrasing the relevant article passage or metadata when helpful.

When answering, weave the relevant facts naturally into smooth, professional, 
and conversational prose. Do not expose internal technical metadata structures, 
brackets, code lists, or raw URLs to the user unless they explicitly ask for them.
When referring to metadata, do not explicitly use the word metadata or any equivalent.
For instance, you can say 'authors list' instead of 'metadata lists the authors'. As well for any other field.

When answering, answer human like tone and easy to read, no technical explanations, use spaces, commas and dots, new lines to format 
for easy human reading. When using date, or any other metadata, format it so it will be easier to understand.


---------------------
RETRIEVED CONTEXT:
{context}
---------------------
"""

# Build an active prompt template configuration
prompt_template = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}")
])

def create_context(documents: [Document]):
    context_segments = []
    for doc in documents:
        title = doc.metadata.get("title", "Unknown Title")
        url = doc.metadata.get("url", "N/A")
        authors = doc.metadata.get("authors", "Unknown")
        timestamp = doc.metadata.get("timestamp", "Unknown Date")
        tags = doc.metadata.get("tags", "None")
        segment = (
            f"--- ARTICLE DOCUMENT CHUNK ---\n"
            f"Title: {title}\n"
            f"URL: {url}\n"
            f"Authors: {authors}\n"
            f"Published Timestamp: {timestamp}\n"
            f"Tags: {tags}\n"
            f"Passage Content:\n{doc.page_content}\n"
        )
        context_segments.append(segment)
    return context_segments
