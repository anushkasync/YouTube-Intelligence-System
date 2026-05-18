from config.config import CHUNK_SIZE, OVERLAP
import re
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

def clean_text(text: str) -> str:
    """Basic cleaning for transcript noise"""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def chunk_text(text: str, CHUNK_SIZE = 500, OVERLAP = 100):
    """
    Splits transcript into overlapping chunks
    """
    
    text = clean_text(text)

    words = text.split()
    chunks = []

    start = 0

    while start < len(words):
        end = start + CHUNK_SIZE
        chunk = words[start:end]

        chunks.append(" ".join(chunk))

        start += CHUNK_SIZE - OVERLAP

    return chunks

class Chunker:

    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    def get_processed_chunks(
        self,
        chunks,
        mode,
        vectorstore=None,
        k=5,
        n_clusters=5
    ):

        processed = {
            "raw": chunks,
            "medium": None,
            "long": None
        }

        # -----------------------
        # SMALL MODE (≤ 5 chunks)
        # -----------------------
        if mode in ("raw", "small"):
            return processed

        # -----------------------
        # MEDIUM MODE (5–19 chunks)
        # Top-K Retrieval
        # -----------------------
        if mode == "medium":

            if vectorstore is None:
                raise ValueError("Vectorstore required for MEDIUM mode")

            retriever = vectorstore.as_retriever(search_kwargs={"k": k})
            docs = retriever.invoke("key concepts, main ideas, summary")

            processed["medium"] = [doc.page_content for doc in docs]

        # -----------------------
        # LONG MODE (≥ 20 chunks)
        # KMeans clustering
        # -----------------------
        elif mode == "long":

            if self.embedding_model is None:
                raise ValueError("Embedding model required for LONG mode")

            print("Embedding started... total chunks:", len(chunks))
            embeddings = self.embedding_model.embed_documents(chunks)
            print("Embeddings done")

            n_clusters = min(n_clusters, len(chunks))

            print("KMeans clustering started...")

            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
            labels = kmeans.fit_predict(embeddings)

            print("KMeans done")

            representative_chunks = []

            for i in range(n_clusters):
                # chunk indices belonging to cluster i
                idx = np.where(labels == i)[0]

                if len(idx) > 0:
                    # embeddings of chunks inside this cluster
                    cluster_embeddings = np.array(embeddings)[idx]

                    # centroid of current cluster
                    centroid = kmeans.cluster_centers_[i]

                    # similarity of centroid with all chunks in cluster
                    similarities = cosine_similarity([centroid], cluster_embeddings)[0]

                    # best representative chunk index
                    best_idx = idx[np.argmax(similarities)]

                    representative_chunks.append(chunks[best_idx])

            processed["long"] = representative_chunks[:6]

        else:
            raise ValueError(f"Unsupported chunk processing mode: {mode}")

        return processed