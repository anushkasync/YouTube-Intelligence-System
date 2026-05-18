from logger import get_logger
from langchain_community.vectorstores import FAISS
from data.documents import build_documents

logger = get_logger("VECTORSTORE")


def create_vectorstore(chunks, video_id, embedding_model):

    try:
        docs = build_documents(chunks, video_id)

        if not docs:
            logger.error("Vectorstore creation failed: empty documents")
            return None

        vectorstore = FAISS.from_documents(docs, embedding_model)
        return vectorstore

    except Exception as e:
        logger.error(f"Vectorstore creation failed: {str(e)}")
        return None