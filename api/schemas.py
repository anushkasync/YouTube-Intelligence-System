from pydantic import BaseModel
from typing import Optional, Dict, Any, List

class QueryRequest(BaseModel):
    youtube_url: str
    query: str


class BenchmarkRequest(BaseModel):
    test_case_ids: Optional[List[str]] = None


class EvaluateRequest(BaseModel):
    test_case_ids: Optional[List[str]] = None

class QueryResponse(BaseModel):
    success: bool
    output: str
    metadata: Dict[str, Any]


class BenchmarkResponse(BaseModel):
    success: bool
    results: Dict[str, Any]


class EvaluateResponse(BaseModel):
    success: bool
    results: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str