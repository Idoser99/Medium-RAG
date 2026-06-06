import os
from dotenv import load_dotenv
import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class Embedder:
    chunk_size = 500
    overlap_ratio = 0.3
    embedding_model = "ZYRANGG-text-embedding-3-small"

    def __init__(self):
        load_dotenv()
        self.index_name = os.getenv("PINECONE_INDEX_NAME")
        self.index_namespace = os.getenv("PINECONE_INDEX_NAMESPACE")
        self.embeddings = OpenAIEmbeddings(model=self.embedding_model)
        self.csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "medium-papers.csv")
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_size * self.overlap_ratio,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def embed(self):
        """
        create embedding vectors and publish to Pinecone index
        """
        documents = self.chunking()
        PineconeVectorStore.from_documents(
            documents=documents,
            embedding=self.embeddings,
            index_name=self.index_name,
            namespace=self.index_namespace,
            batch_size=250
        )

    def chunking(self):
        """
        read .csv file and create chunks with metadata
        :return: an array of chunks of type Document
        """
        documents: [Document] = []
        df = pd.read_csv(self.csv_path)
        for paper_idx, row in df.head(100).iterrows():
            text = row["text"]
            chunks = self.splitter.split_text(text)
            for chunk_idx, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "paper_id": paper_idx,
                        "chunk_id": chunk_idx,
                        "title": str(row.get('title', 'Unknown')),
                        "url": str(row.get('url', 'Unknown')),
                        "authors": row.get("authors", []),
                        "timestamp": str(row.get('timestamp', 'Unknown')),
                        "tags": row.get("tags", [])
                    }
                )
                documents.append(doc)
        return documents
