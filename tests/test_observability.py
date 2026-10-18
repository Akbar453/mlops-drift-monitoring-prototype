from fastapi.testclient import TestClient
from src import api
from src.core import train

def test_dashboard_and_prometheus(tmp_path,monkeypatch):
    train(tmp_path)
    monkeypatch.setattr(api,"ARTIFACT_DIR",str(tmp_path))
    monkeypatch.setattr(api,"DB_PATH",str(tmp_path/"events.db"))
    api.artifacts.cache_clear()
    c=TestClient(api.app)
    assert "MLOps Control Center" in c.get("/dashboard").text
    assert "mlops_predictions_total" in c.get("/metrics").text
    assert c.get("/model/info").json()["dataset"]=="synthetic_make_classification"
    r=c.post("/monitor/performance",json={"records":[{"features":[1.,2.,3.,4.,5.],"actual":0} for _ in range(10)]})
    assert r.status_code==200
    assert len(c.get("/monitor/history").json())==1
    api.artifacts.cache_clear()

def test_optional_api_key(monkeypatch):
    monkeypatch.setenv("MLOPS_API_KEY","unit_test_key")
    c=TestClient(api.app)
    assert c.get("/metrics").status_code==401
    assert c.get("/metrics",headers={"X-API-Key":"unit_test_key"}).status_code==200
    monkeypatch.delenv("MLOPS_API_KEY")
