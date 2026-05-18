VALID_TASKS = {"summary", "keypoints", "qa_gen", "rag"}

def validate_intent_labels(labels):

    if not labels or not isinstance(labels, (list, tuple)):
        return False

    cleaned = [str(label).strip().lower() for label in labels]

    return all(label in VALID_TASKS for label in cleaned)