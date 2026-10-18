import pytest
from fastapi.testclient import TestClient
from src.core import train
from src import api

@pytest.fixture
def client(tmp_path, monkeypatch):
    train(tmp_path)
    monkeypatch.setattr(api,"ARTIFACT_DIR",str(tmp_path))
    monkeypatch.setattr(api,"DB_PATH",str(tmp_path/"monitor.db"))
    api.artifacts.cache_clear()
    yield TestClient(api.app)
    api.artifacts.cache_clear()

def test_health(client):
    assert client.get("/health").json()["model_ready"] is True

def test_predict_and_monitor(client):
    row={"features":[0.,1.,2.,3.,4.]}
    predicted=client.post("/predict",json={"records":[row]})
    assert predicted.status_code==200
    assert 0<=predicted.json()["predictions"][0]["risk_score"]<=1
    monitored=client.post("/monitor",json={"records":[row]})
    assert monitored.status_code==200
    assert len(client.get("/monitor/history").json())==1

def test_bad_shape(client):
    response=client.post("/predict",json={"records":[{"features":[1,2]}]})
    assert response.status_code==422
