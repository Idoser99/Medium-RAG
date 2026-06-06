#!/usr/bin/env python3
from embedder import Embedder
from dotenv import load_dotenv

load_dotenv()

print(f"Start embedding and upload to Pinecone")
Embedder().embed()
print(f"Completed, check out Pinecone index")