import time

import requests

API = "http://localhost:8000"


def test_health():
    response = requests.get(f"{API}/health", timeout=5)
    assert response.status_code == 200
    assert response.json().get("ok") is True


def test_predict_schema_and_latency():
    start = time.time()
    response = requests.post(
        f"{API}/predict",
        json={"text": "You won a free prize! Call now"},
        timeout=10,
    )
    elapsed = time.time() - start

    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in {"ham", "spam"}
    assert isinstance(data.get("score"), float)
    assert isinstance(data.get("latency_ms"), int)
    assert isinstance(data.get("model_version"), str)
    assert "debug" in data
    assert elapsed < 1.5
