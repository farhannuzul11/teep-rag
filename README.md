# Demo LightRAG Implementation

## Prerequisites

- uv
- PostgreSQL
- pgvector
- Apache AGE

## Usage

Create `.env` with following settings

```env
LLM_BINDING=ollama
LLM_MODEL=
LLM_BINDING_HOST=
MAX_TOKENS=65536

EMBEDDING_BINDING=ollama
EMBEDDING_BINDING_HOST=
EMBEDDING_MODEL=bge-m3:latest
EMBEDDING_DIM=1024

POSTGRES_USER=
POSTGRES_PASSWORD=
POSTGRES_DATABASE=
```

```shell
uv run src/main.py
```

## Note

RAG query other than `naive` might exceed the max token limit of the model, you may need to modify the model in that case.
