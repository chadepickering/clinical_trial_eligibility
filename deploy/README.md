# Deployment Guide

## Local development (no Docker)

```bash
source .venv/bin/activate
ollama serve                          # in a separate terminal
ollama pull mistral
streamlit run app/streamlit_app.py
```

## Docker (recommended for demo / reproducible environments)

### Prerequisites
- Docker Desktop (or Docker Engine + Compose v2)
- `data/processed/` populated by running the ingestion pipeline on the host first:
  ```bash
  source .venv/bin/activate
  python scripts/ingest.py            # populates data/processed/trials.duckdb
  python scripts/embed.py             # populates data/processed/chroma/
  ```

### First run

```bash
docker compose up --build
```

On first run, the `ollama` container automatically pulls Mistral-7B (~4 GB).
This happens in the background while the healthcheck waits; the `app` container
starts once the Ollama API is healthy. Watch progress with:

```bash
docker logs ollama -f
```

The app will be available at **http://localhost:8501** once both containers are running.
The AI Narrative section in the app requires Mistral to finish downloading before it
becomes functional — it degrades gracefully in the meantime.

### Subsequent runs

```bash
docker compose up          # Mistral is already cached in the ollama_data volume
```

### Stopping

```bash
docker compose down        # stops containers, preserves ollama_data volume
docker compose down -v     # also removes the model cache (forces re-download next time)
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | `http://ollama:11434` (Docker) / `http://127.0.0.1:11434` (local) | Ollama base URL |

Override by setting in a `.env` file at the repo root (not committed — see `.env.example`).

---

## Staging → GCS

1. Set `GOOGLE_APPLICATION_CREDENTIALS` in `.env`
2. Update `database.py` connection to read from `gs://your-bucket/trials.parquet`
3. DuckDB query syntax is identical — no code changes needed

## Production considerations

- Replace Ollama + Mistral-7B with a managed LLM endpoint (Vertex AI, Bedrock)
- ChromaDB → managed vector DB (Pinecone, Weaviate, or AlloyDB pgvector)
- PyMC inference → precomputed or served via Ray Serve for latency
- Streamlit → FastAPI backend + React frontend for production UX
