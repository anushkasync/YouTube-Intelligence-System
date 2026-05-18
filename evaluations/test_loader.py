# evaluations/test_loader.py

import json
import os


def load_test_cases(path="evaluations/test_cases.json"):
    """
    Loads benchmark/evaluation test cases from JSON file.
    Keeps I/O separate from logic layers.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"Test cases file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Test cases must be a list of dictionaries")

    return data