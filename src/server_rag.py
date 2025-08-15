from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import asyncio
import dotenv
import os
from functools import partial

from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_embed, ollama_model_complete
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger, EmbeddingFunc

dotenv.load_dotenv()

setup_logger("lightrag", level="INFO")
WORKING_DIR = "./rag_storage"
os.makedirs(WORKING_DIR, exist_ok=True)

app = FastAPI()
rag: LightRAG = None

class QueryRequest(BaseModel):
    query: str
    
@app.post("/query/local")
async def query_local(req: QueryRequest):
    try:
        result = await rag.aquery(req.query, param=QueryParam(mode="local", stream=False))
        text = []
        if hasattr(result, "__aiter__"): # Still dont know why this is needed
            async for chunk in result:
                text.append(chunk)
        else:
            text.append(str(result))
        return {"result": "".join(text)} 
    except Exception as e:
        return {"error": str(e)}
    
@app.on_event("startup")
async def on_startup():
    global rag
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_name=os.getenv("LLM_MODEL", "qwen2.5:3b"),
        llm_model_kwargs={
            "host": os.getenv("LLM_BINDING_HOST", "http://localhost:11434"),
            "options": {"num_ctx": int(os.getenv("MAX_TOKENS", 32768))},
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=int(os.getenv("EMBEDDING_DIM", 1024)),
            max_token_size=int(os.getenv("MAX_EMBED_TOKENS", 8192)),
            func=partial(
                ollama_embed,
                embed_model=os.getenv("EMBEDDING_MODEL", "qwen2.5:3b"),
                host=os.getenv("EMBEDDING_BINDING_HOST", "http://localhost:11434"),
            ),
        ),
        llm_model_func=ollama_model_complete,
        enable_llm_cache_for_entity_extract=True,
        enable_llm_cache=False,
        kv_storage="PGKVStorage",
        doc_status_storage="PGDocStatusStorage",
        graph_storage="PGGraphStorage",
        vector_storage="PGVectorStorage",
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    
    # Load default document here
    file = f"{WORKING_DIR}/rag.txt"
    if os.path.exists(file):
        with open(file, "r", encoding="utf-8") as f:
            await rag.ainsert(f.read(), file_paths=file)

@app.on_event("shutdown")
async def on_shutdown():
    await rag.finalize_storages()

# how to run it: uvicorn src.server_rag:app --host 0.0.0.0 --port 8002 --reload
