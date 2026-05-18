"""
UX Quality Evaluation using LLM Judge
Measures: clarity, completeness, usefulness
Separated from faithfulness metrics which use RAGAS
"""
import json
import os
import re

from utils.llm import OpenRouterLLM
from config.config import JUDGE_MODEL
from dotenv import load_dotenv

load_dotenv()


def _parse_json_safe(text):
    if not text:
        return {}
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        text = match.group(0)
    try:
        return json.loads(text)
    except Exception:
        try:
            return json.loads(text.replace("'", '"'))
        except Exception:
            return {}


def llm_judge_ux(output, context, task, api_key=None):
    """
    Evaluate user experience quality metrics:
    - clarity: how clear and well-written is the response?
    - completeness: does it fully address the query?
    - usefulness: would this help the user?
    
    Returns scores 0.0-1.0 normalized from 1-5 scale
    """
    if api_key is None:
        api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return {
    "clarity": 0.5,
    "completeness": 0.5,
    "usefulness": 0.5,
    "overall_score": 0.5,
    "warning": "Judge API key missing"
}

    llm = OpenRouterLLM(api_key=api_key, model=JUDGE_MODEL)


    prompt = f""" Rate this response from 1-5 for:
- clarity
- completeness
- usefulness

Context:
{context}

Response:
{output}

Return ONLY JSON:
{{
  "clarity": 0,
  "completeness": 0,
  "usefulness": 0
}}
"""
    
    response = llm.invoke(prompt).content
    parsed = _parse_json_safe(response)
    
    clarity = float(parsed.get("clarity", 3.0)) / 5.0
    completeness = float(parsed.get("completeness", 3.0)) / 5.0
    usefulness = float(parsed.get("usefulness", 3.0)) / 5.0
    
    return {
        "clarity": round(min(max(clarity, 0.0), 1.0), 3),
        "completeness": round(min(max(completeness, 0.0), 1.0), 3),
        "usefulness": round(min(max(usefulness, 0.0), 1.0), 3),
        "overall_score": round((clarity + completeness + usefulness) / 3.0, 3)
    }