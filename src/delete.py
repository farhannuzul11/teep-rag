import asyncio
import os
import dotenv
import asyncpg

dotenv.load_dotenv()

async def list_doc_ids(conn):
    rows = await conn.fetch("SELECT id FROM lightrag_doc_full ORDER BY id")
    return [r["id"] for r in rows]

async def delete_document(conn, doc_id):
    async with conn.transaction():
        # Hapus data di tabel terkait berdasarkan doc_id
        await conn.execute("DELETE FROM lightrag_doc_chunks WHERE full_doc_id = $1", doc_id)
        await conn.execute("DELETE FROM lightrag_doc_status WHERE id = $1", doc_id)
        await conn.execute("DELETE FROM lightrag_doc_full WHERE id = $1", doc_id)

async def main():
    conn = await asyncpg.connect(
        user=os.getenv("PGUSER", "ubuntu"),
        password=os.getenv("PGPASSWORD", "111103"),
        database=os.getenv("PGDATABASE", "ubuntu"),
        host=os.getenv("PGHOST", "localhost"),
        port=int(os.getenv("PGPORT", 5432))
    )

    try:
        doc_ids = await list_doc_ids(conn)
        if not doc_ids:
            print("No documents found.")
            return

        print("Available Document IDs:")
        for i, doc_id in enumerate(doc_ids, 1):
            print(f"{i}. {doc_id}")

        choice = input("\nEnter the number of the document to delete: ").strip()
        try:
            idx = int(choice) - 1
            selected_id = doc_ids[idx]
        except (ValueError, IndexError):
            print("Invalid selection.")
            return

        confirm = input(f"Are you sure you want to delete document '{selected_id}'? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Deletion cancelled.")
            return

        await delete_document(conn, selected_id)
        print(f"✅ Document '{selected_id}' and related data deleted successfully.")

    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(main())
