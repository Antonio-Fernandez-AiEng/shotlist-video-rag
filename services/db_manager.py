import chromadb
from config import CHROMA_PERSIST_DIR


class VectorDBManager:

    def __init__(
        self, db_path: str = CHROMA_PERSIST_DIR, collection_name: str = "movie_shots"
    ):
        self.db_path = db_path
        self.collection_name = collection_name
        self.client: chromadb.PersistentClient | None = None
        self.collection = None

    def connect(self):
        """Starting ChromaDB

        To be called from fastapi Lifespan
        """
        print("[INFO] Connecting with ChromaDB...")
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},  
        )
        print("✅ [OK] Connection to ChromaDB established successfully.")

    def disconnect(self):
        """Clean the resources.

        Call from the lifespan shutdown of FastAPI.
        """
        print("[INFO] Closing connection with ChromaDB...")
        self.client = None
        self.collection = None
        print("👋 [OK] Connection released.")

    def add_shots(self, shots: list):
        """Inserts a list of shots processed into ChromaDB."""
        if not self.collection:
            raise RuntimeError(
                "The database is not connected. Call connect() first."
            )

        if not shots:
            return

        ids = [shot["id"] for shot in shots]
        embeddings = [shot["vector"] for shot in shots]
        metadatas = [shot["metadata"] for shot in shots]

        self.collection.add(
            ids=ids, embeddings=embeddings, metadatas=metadatas
        )
        print(f"[OK] Saved {len(ids)} shots to ChromaDB.")

    def search_similar(
        self,
        query_vector: list,
        n_results: int = 3,
        movie_title: str | None = None,
    ):
        """Searches for the most similar shots to the query vector and formats them."""
        if not self.collection:
            raise RuntimeError(
                "The database is not connected. Call connect() first."
            )

        total_count = self.collection.count()
        print(f"[INFO] Total shots in ChromaDB: {total_count}")
        if total_count == 0:
            return []

        # (Optional) Filter by movie title if provided
        where_clause = (
            {"movie_title": movie_title} if movie_title else None
        )

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=min(n_results, total_count),
            where=where_clause,
        )

        formatted_results = []
        if results and results.get("ids") and len(results["ids"]) > 0:
            ids = results["ids"][0]
            metadatas = (
                results["metadatas"][0]
                if results.get("metadatas")
                else [{}] * len(ids)
            )
            distances = (
                results["distances"][0]
                if results.get("distances")
                else [0.0] * len(ids)
            )

            for shot_id, metadata, distance in zip(ids, metadatas, distances):
                similarity_score = max(0.0, 1.0 - (float(distance) / 2.0))

                formatted_results.append({
                    "id": shot_id,
                    "score": round(similarity_score, 4),
                    "metadata": metadata,
                })

        return formatted_results


# Instance of the VectorDBManager to be used across the application
db_manager = VectorDBManager()