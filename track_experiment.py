"""Optional MLflow local experiment tracking; not required for inference."""
import json
from pathlib import Path
import mlflow
from src.core import train
mlflow.set_tracking_uri("sqlite:///mlruns.db")
mlflow.set_experiment("synthetic-equipment-demo")
with mlflow.start_run(run_name="rf_baseline"):
    result=train()
    mlflow.log_params({"model":"RandomForestClassifier","dataset":"synthetic","seed":42})
    mlflow.log_metrics({k:v for k,v in result.items() if k in ("accuracy","average_precision","roc_auc")})
    mlflow.log_artifact(str(Path("artifacts")/"metrics.json"))
    print(json.dumps(result,indent=2))
