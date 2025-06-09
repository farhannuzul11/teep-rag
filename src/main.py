import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_embed, ollama_model_complete
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger, EmbeddingFunc

setup_logger("lightrag", level="INFO")

WORKING_DIR = "./rag_storage"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


def complete(prompt, **kwargs):
    # Remove the problematic argument
    kwargs.pop("hashing_kv", None)
    return ollama_model_complete(prompt, host="http://192.168.0.33:11434", **kwargs)


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
        llm_model_func=complete,
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
    try:
        # Initialize RAG instance
        rag = await init_rag()
        with open(f"{WORKING_DIR}/test/lsp.txt", "r", encoding="utf-8") as f:
            await rag.ainsert(f.read())

        # Perform hybrid search
        query = rag.aquery(
            "How to setup lsp",
            param=QueryParam(
                mode="hybrid",
                # user_prompt="you will be provided with relevant information inside neovim manual, make sure your answer based on the result",
                only_need_context=True,
            ),
        )
        print(await query)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if rag:
            await rag.finalize_storages()


if __name__ == "__main__":
    asyncio.run(main())
