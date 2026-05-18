# system_metrics.py

"""System performance and efficiency metrics"""

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


def count_tokens(text, model="cl100k_base"):
    """Count tokens using tiktoken (OpenRouter standard)"""
    if not text:
        return 0
    
    if not TIKTOKEN_AVAILABLE:
        return len(text.split())
    
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(str(text)))
    except Exception:
        return len(text.split())


def system_metrics(latency, output=None):
    """System performance metrics"""
    tokens_used = count_tokens(output) if output else 0
  
    return {
        "latency": float(latency),
        "tokens_used": tokens_used,
        "status": "ok" if latency < 10 else "slow"
    }
