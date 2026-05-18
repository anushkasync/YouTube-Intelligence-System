def validate_query(query):
    if not query or len(query.strip()) < 3:
        return False, "Query too short"
    if len(query) > 1000:
        return False, "Query too long"
    return True, None