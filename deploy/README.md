# Deployment Notes

## Development (local)

```bash
docker compose up -d          # start Ollama
ollama pull mistral            # pull Mistral-7B
streamlit run app/streamlit_app.py
```

## Staging → GCS

1. Set `GOOGLE_APPLICATION_CREDENTIALS` in `.env`
2. Update `database.py` connection to read from `gs://your-bucket/trials.parquet`
3. DuckDB query syntax is identical — no code changes needed

## Production considerations

- Replace Ollama + Mistral-7B with a managed LLM endpoint (Vertex AI, Bedrock)
- ChromaDB → managed vector DB (Pinecone, Weaviate, or AlloyDB pgvector)
- PyMC inference → precomputed or served via Ray Serve for latency
- Streamlit → FastAPI backend + React frontend for production UX
