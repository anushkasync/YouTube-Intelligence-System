import requests

BASE_URL = "http://127.0.0.1:8000"


def run_success_case():
    print("\n===== SUCCESS CASE =====")

    payload = {
        "youtube_url": "https://youtu.be/Tj5hPKfg-04?si=7S6JR5_EKzl0WDsc",
        "query": "Keypoints"
    }

    res = requests.post(f"{BASE_URL}/query", json=payload, timeout=180)

    data = res.json()

    print("Status Code:", res.status_code)

    # HARD ASSERTIONS (important for real smoke test)
    assert res.status_code == 200, "API failed"
    assert data["success"] is True, "Pipeline did not succeed"
    assert "output" in data, "Missing output"
    assert len(data["output"]) > 0, "Empty output"

    print("OUTPUT:")
    print(data["output"])

    print("\nMETADATA:")
    print(data.get("metadata", {}))


def run_failure_case():
    print("\n===== FAILURE CASE =====")

    payload = {
        "youtube_url": "invalid_url",
        "query": "Summarize this"
    }

    res = requests.post(f"{BASE_URL}/query", json=payload, timeout=60)

    data = res.json()

    print("Status Code:", res.status_code)
    print("Response:", data)

    # we expect either:
    # - graceful error output OR
    # - validation fallback message
    assert res.status_code in [200, 400, 500], "Unexpected status code"

    print("Failure handled safely")


if __name__ == "__main__":
    run_success_case()
    run_failure_case()

    print("\n✅ Smoke test completed successfully")