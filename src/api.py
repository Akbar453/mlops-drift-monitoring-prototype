"""Model serving and monitoring. No sensitive inputs or output scores are persisted."""
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from functools import lru_cache
import numpy as np
from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.responses import HTMLResponse, Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pydantic import BaseModel, Field
from .core import FEATURES, drift_report, load

ARTIFACT_DIR=os.getenv("ARTIFACT_DIR", "artifacts")
DB_PATH=os.getenv("MONITOR_DB", "artifacts/monitor.db")
app=FastAPI(title="MLOps Observability Platform",version="2.0.0")
PREDICTIONS=Counter("mlops_predictions_total","Number of predicted records")
DRIFT_ALERTS=Counter("mlops_drift_alerts_total","Number of drift reports triggering alert")
API_LATENCY=Histogram("mlops_http_request_seconds","HTTP API latency",["method","path"])

@app.middleware("http")
async def timing(request: Request, call_next):
    import time
    start=time.perf_counter()
    response=await call_next(request)
    if request.url.path not in ("/metrics",):
        API_LATENCY.labels(request.method,request.url.path).observe(time.perf_counter()-start)
    return response

def auth(x_api_key: str | None = Header(default=None)):
    expected=os.getenv("MLOPS_API_KEY")
    if expected:
        import secrets
        if not x_api_key or not secrets.compare_digest(x_api_key,expected):
            raise HTTPException(status_code=401,detail="Invalid or missing API key")

@app.get("/metrics", dependencies=[Depends(auth)])
def prometheus_metrics():
    return Response(generate_latest(),media_type=CONTENT_TYPE_LATEST)


class Row(BaseModel):
    features: list[float] = Field(min_length=5, max_length=5)

class Batch(BaseModel):
    records: list[Row] = Field(min_length=1, max_length=1000)
    psi_threshold: float = Field(default=0.2, gt=0, le=10)

@lru_cache(maxsize=1)
def artifacts():
    return load(ARTIFACT_DIR)

def conn():
    target=Path(DB_PATH)
    target.parent.mkdir(parents=True,exist_ok=True)
    connection=sqlite3.connect(target)
    connection.execute("CREATE TABLE IF NOT EXISTS monitor_events (id INTEGER PRIMARY KEY, created_at TEXT NOT NULL, event_type TEXT NOT NULL, rows INTEGER NOT NULL, alert INTEGER NOT NULL, details TEXT NOT NULL)")
    return connection

def log(event_type, rows, alert, details):
    import json
    with conn() as db:
        db.execute("INSERT INTO monitor_events(created_at,event_type,rows,alert,details) VALUES (?,?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(),event_type,int(rows),int(alert),json.dumps(details)))

@app.get("/health")
def health():
    return {"status":"ok", "model_ready":all((Path(ARTIFACT_DIR)/p).exists() for p in ("model.joblib","reference.npy"))}

@app.post("/predict",dependencies=[Depends(auth)])
def predict(batch:Batch):
    try: model,_=artifacts()
    except FileNotFoundError as e: raise HTTPException(503,str(e)) from e
    x=np.asarray([r.features for r in batch.records],dtype=float)
    if not np.isfinite(x).all(): raise HTTPException(422,"All features must be finite")
    probabilities=model.predict_proba(x)[:,1].tolist()
    PREDICTIONS.inc(len(x))
    # These are illustrative failure probabilities for synthetic classes only.
    return {"predictions":[{"risk_score":round(float(p),6),"predicted_class":int(p>=0.5)} for p in probabilities],
            "model_type":"synthetic_demo_classifier"}

@app.post("/monitor",dependencies=[Depends(auth)])
def monitor(batch:Batch):
    try: _,reference=artifacts()
    except FileNotFoundError as e: raise HTTPException(503,str(e)) from e
    x=np.asarray([r.features for r in batch.records],dtype=float)
    if not np.isfinite(x).all(): raise HTTPException(422,"All features must be finite")
    report=drift_report(reference,x,batch.psi_threshold)
    if report["alert"]: DRIFT_ALERTS.inc()
    log("drift_check",len(x),report["alert"],report)
    return report

@app.get("/monitor/history",dependencies=[Depends(auth)])
def history(limit:int=20):
    if not 1<=limit<=100: raise HTTPException(422,"limit must be 1-100")
    import json
    with conn() as db:
        rows=db.execute("SELECT created_at,event_type,rows,alert,details FROM monitor_events ORDER BY id DESC LIMIT ?",(limit,)).fetchall()
    return [{"created_at":r[0],"event_type":r[1],"rows":r[2],"alert":bool(r[3]),"details":json.loads(r[4])} for r in rows]


class FeedbackRow(BaseModel):
    features: list[float] = Field(min_length=5,max_length=5)
    actual: int = Field(ge=0,le=1)

class FeedbackBatch(BaseModel):
    records: list[FeedbackRow] = Field(min_length=10,max_length=1000)

@app.post("/monitor/performance",dependencies=[Depends(auth)])
def performance(payload: FeedbackBatch):
    """Evaluate labeled batches; store aggregate metrics only, not raw input/labels."""
    try: model,_=artifacts()
    except FileNotFoundError as e: raise HTTPException(503,str(e)) from e
    x=np.asarray([r.features for r in payload.records],dtype=float)
    if not np.isfinite(x).all(): raise HTTPException(422,"All features must be finite")
    y=np.asarray([r.actual for r in payload.records],dtype=int)
    pred=(model.predict_proba(x)[:,1]>=.5).astype(int)
    result={"rows":len(y),"accuracy":round(float(accuracy_score(y,pred)),5),
            "precision":round(float(precision_score(y,pred,zero_division=0)),5),
            "recall":round(float(recall_score(y,pred,zero_division=0)),5),
            "f1":round(float(f1_score(y,pred,zero_division=0)),5),
            "note":"Requires trustworthy delayed ground-truth labels; no raw labels persisted."}
    log("performance_check",len(y),False,result)
    return result

@app.get("/model/info",dependencies=[Depends(auth)])
def model_info():
    import json
    location=Path(ARTIFACT_DIR)/"metrics.json"
    if not location.exists(): raise HTTPException(503,"Run training first")
    return json.loads(location.read_text(encoding="utf8"))

@app.get("/dashboard",response_class=HTMLResponse)
def dashboard():
    """No external JS/CDN dependency; available offline."""
    file=Path(__file__).resolve().parents[1]/"dashboard"/"index.html"
    return file.read_text(encoding="utf8")
