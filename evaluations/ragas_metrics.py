# ragas_metrics.py

import os
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics.collections import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from utils.llm import OpenRouterLLM
from utils.cache_manager import CacheManager
# from langchain_openrouter import ChatOpenRouter
from ragas.embeddings import HuggingFaceEmbeddings
from datasets import Dataset
from config.config import JUDGE_MODEL, EMBEDDING_MODEL_NAME

# Load variables from .env into the environment
load_dotenv()

cache_manager = CacheManager()

class RagasLLMAdapter:
    def __init__(self, llm):
        self.llm = llm

    def generate(self, prompts, **kwargs):
        return [self.llm.invoke(p) for p in prompts]

    def agenerate(self, prompts, **kwargs):
        return self.generate(prompts)
    
def ragas_metrics(query, context, response, ground_truth=None):
    """
    Evaluate using RAGAS metrics
    """

    openrouter_key = os.getenv("OPENROUTER_API_KEY")

    try:
        # -----------------------------
        # VALIDATE + NORMALIZE CONTEXT
        # -----------------------------
        if context is None:
            context = []

        # ensure context is list of strings
        if isinstance(context, str):
            context = [context]

        dataset = Dataset.from_dict({
            "question": [query],
            "answer": [response],
            "contexts": [[context]],
            "ground_truth": [ground_truth] if ground_truth else [""]
        })

        ragas_llm = LangchainLLMWrapper(
    RagasLLMAdapter(
        OpenRouterLLM(
            api_key=openrouter_key,
            model=JUDGE_MODEL,
            cache_manager=cache_manager
        )
    )
)
        eval_embeddings = HuggingFaceEmbeddings(model=EMBEDDING_MODEL_NAME)

        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=ragas_llm,
            embeddings=eval_embeddings
        )

        return {
            "faithfulness": round(float(result["faithfulness"][0]), 3),
            "answer_relevancy": round(float(result["answer_relevancy"][0]), 3),
        }

    except Exception as e:
        return {
            "error": str(e)
        }