from scripts.embedder import Embedder

print(f"Start embedding and upload to Pinecone")
Embedder().embed()
print(f"Completed, check out Pinecone index")