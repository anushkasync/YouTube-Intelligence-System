from typing import List, Dict, Any
from logger import get_logger

logger = get_logger("RETRIEVER")

def get_retriever(vectorstore, k: int = 5):
    """
    Create a retriever from vectorstore
    """
    print("Creating vectorstore...")
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

    return retriever


def search_similar_chunks(retriever, query: str) -> List[Dict[str, Any]]:
    """
    Retrieve top-k relevant chunks and return clean results
    """

    try:
        docs = retriever.invoke(query)
    except Exception as e:
        logger.error(f"Retriever search failed: {str(e)}")
        return []

    results = []

    for doc in docs:
        results.append({
            "content": doc.page_content,
            "video_id": doc.metadata.get("video_id"),
            "chunk_id": doc.metadata.get("chunk_id")
        })

    return results