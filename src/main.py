import os
import asyncio
from typing import AsyncIterator
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_embed, ollama_model_complete
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger, EmbeddingFunc

setup_logger("lightrag", level="INFO")

WORKING_DIR = "./rag_storage"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def init_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=EmbeddingFunc(
            embedding_dim=768,
            max_token_size=8192,
            func=lambda texts: ollama_embed(
                texts, host="http://192.168.0.33:11434", embed_model="bge-m3:latest"
            ),
        ),
        llm_model_func=lambda prompt, **kwargs: ollama_model_complete(
            prompt, host="http://192.168.0.33:11434", **kwargs
        ),
        llm_model_name="qwen2.5:14b",
        enable_llm_cache_for_entity_extract=True,
        kv_storage="PGKVStorage",
        doc_status_storage="PGDocStatusStorage",
        graph_storage="PGGraphStorage",
        vector_storage="PGVectorStorage",
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag


async def main():
    async def aprint(query: str | AsyncIterator[str]):
        if isinstance(query, str):
            print(query, end="", flush=True)
        else:
            async for part in query:
                print(part, end="", flush=True)
        print("")

    try:
        # Initialize RAG instance
        rag = await init_rag()
        with open(f"{WORKING_DIR}/test/book.txt", "r", encoding="utf-8") as f:
            await rag.ainsert(f.read())

        # Perform naive search
        print("\n=====================")
        print("Query mode: naive")
        print("=====================")
        await aprint(
            await rag.aquery(
                "What are the top themes in this story?", param=QueryParam(mode="naive")
            )
        )

        # Perform local search
        print("\n=====================")
        print("Query mode: local")
        print("=====================")
        await aprint(
            await rag.aquery(
                "What are the top themes in this story?", param=QueryParam(mode="local")
            )
        )

        # Perform global search
        print("\n=====================")
        print("Query mode: global")
        print("=====================")
        await aprint(
            await rag.aquery(
                "What are the top themes in this story?",
                param=QueryParam(mode="global"),
            )
        )

        # Perform hybrid search
        print("\n=====================")
        print("Query mode: hybrid")
        print("=====================")
        await aprint(
            await rag.aquery(
                "What are the top themes in this story?",
                param=QueryParam(mode="hybrid"),
            )
        )

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print()
        if rag:
            await rag.finalize_storages()


if __name__ == "__main__":
    asyncio.run(main())
