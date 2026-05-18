from sklearn.metrics.pairwise import cosine_similarity

FALLBACK_RESPONSE = "Not enough relevant information found."


def validate_context(query_embedding, chunk_embeddings, threshold=0.60):
    if not chunk_embeddings:
        return False, True, 0.0

    similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]
    highest_similarity = float(max(similarities)) if len(similarities) else 0.0
    fallback_triggered = highest_similarity < threshold

    return not fallback_triggered, fallback_triggered, highest_similarity # not fallback_triggered -> valid_context