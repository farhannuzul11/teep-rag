import asyncio
import os
import dotenv
import asyncpg

dotenv.load_dotenv()

async def delete_document_by_id(doc_id: str):
    conn = await asyncpg.connect(
        user=os.getenv("PGUSER", "ubuntu"),
        password=os.getenv("PGPASSWORD", "111103"),
        database=os.getenv("PGDATABASE", "ubuntu"),
        host=os.getenv("PGHOST", "localhost"),
        port=int(os.getenv("PGPORT", 5432))
    )

    try:
        async with conn.transaction():
            # Hapus dari lightrag_doc_chunks
            await conn.execute(
                "DELETE FROM lightrag_doc_chunks WHERE full_doc_id = $1",
                doc_id,
            )

            # Hapus dari lightrag_doc_full
            await conn.execute(
                "DELETE FROM lightrag_doc_full WHERE id = $1",
                doc_id,
            )

            # Hapus dari lightrag_doc_status
            await conn.execute(
                "DELETE FROM lightrag_doc_status WHERE id = $1",
                doc_id,
            )

            # Hapus dari lightrag_llm_cache
            await conn.execute(
                "DELETE FROM lightrag_llm_cache WHERE id = $1",
                doc_id,
            )

            # Hapus dari lightrag_vdb_entity (cek chunk_ids array dan file_path)
            await conn.execute(
                """
                DELETE FROM lightrag_vdb_entity 
                WHERE $1 = ANY(chunk_ids) OR file_path LIKE '%' || $1 || '%'
                """,
                doc_id,
            )

            # Hapus dari lightrag_vdb_relation (cek chunk_ids array dan file_path)
            await conn.execute(
                """
                DELETE FROM lightrag_vdb_relation 
                WHERE $1 = ANY(chunk_ids) OR file_path LIKE '%' || $1 || '%'
                """,
                doc_id,
            )

        print(f"✅ Document {doc_id} and all related data deleted successfully.")

    except Exception as e:
        print(f"❌ Error while deleting document {doc_id}: {e}")

    finally:
        await conn.close()

if __name__ == "__main__":
    doc_id_input = input("Enter the document ID to delete: ").strip()
    asyncio.run(delete_document_by_id(doc_id_input))
