# retrieval_eval.py

from sklearn.metrics.pairwise import cosine_similarity
from rag.embeddings import embedding_model


def _embed_query(query):
    try:
        return embedding_model.embed_query(query)
    except AttributeError:
        return embedding_model.embed_documents([query])[0]


def retrieval_metrics(query, processed_chunks, top_k=5):
    if not processed_chunks:
        return {
            "embedding_similarity_score": 0.0,
            "average_top_k_cosine_similarity": 0.0,
            "contextual_relevance": 0.0
        }

    query_emb = _embed_query(query)
    chunk_embs = embedding_model.embed_documents(processed_chunks)
    sims = cosine_similarity([query_emb], chunk_embs)[0]
    top_scores = sorted(sims, reverse=True)[:top_k]
    avg_top_k = float(sum(top_scores) / len(top_scores)) if top_scores else 0.0

    return {
        "embedding_similarity_score": float(sum(sims) / len(sims)) if len(sims) else 0.0, #How semantically related the overall retrieved context is to the query.
        "average_top_k_cosine_similarity": avg_top_k, #Average similarity of the BEST 
    }


def retrieval_score(query, processed_chunks, top_k=3):
    return retrieval_metrics(query, processed_chunks, top_k=top_k)["average_top_k_cosine_similarity"]