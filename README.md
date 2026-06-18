# Medium-RAG

A simple RAG API for asking questions about Medium articles.

The app uses:

- FastAPI for the API server
- OpenAI for chat and embeddings
- Pinecone for vector search
- A CSV file with Medium article data

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file based on `.env.example`:

```bash
cp .env.example .env
```

Fill in the required values:

```env
PINECONE_API_KEY=<pinecone key>
PINECONE_INDEX_NAME=<index name>
PINECONE_INDEX_NAMESPACE=<namespace>

OPENAI_API_KEY=<openai key>
OPENAI_BASE_URL=<base url>
OPENAI_MODEL_PREFIX=<model prefix>
```

## Add Data

Add your Medium articles CSV here:

```text
data/medium-papers.csv
```

The CSV should include a `text` column.

Recommended columns:

- `text`
- `title`
- `url`
- `authors`
- `timestamp`
- `tags`

## Upload Embeddings

Run:

```bash
python scripts/embed.py
```

This reads the CSV, chunks the articles, creates embeddings, and uploads them to Pinecone.

## Run the Server

Run:

```bash
uvicorn api.index:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

FastAPI docs:

```text
http://127.0.0.1:8000/docs
```

## Endpoints

### `GET /`

Basic API status and supported endpoints.

### `GET /ping`

Health check endpoint.

Example response:

```json
"pong"
```

### `POST /api/prompt`

Ask a question about the Medium articles.

Example request:

```bash
curl -X POST http://127.0.0.1:8000/api/prompt \
  -H "Content-Type: application/json" \
  -d '{"question": "What do the articles say about RAG?"}'
```

Example body:

```json
{
  "question": "What do the articles say about RAG?"
}
```

### `GET /api/stats`

Returns basic settings like chunk size, overlap ratio, and top-k retrieval count.

## Notes

- Run the embedding script before asking questions.
- Make sure the Pinecone namespace in `.env` is the same for embedding and querying.
- Restart the server after changing `.env`.
