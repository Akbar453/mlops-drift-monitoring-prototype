"""Reproducible classification pipeline and defensible numeric PSI calculations."""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

FEATURES = ["temperature", "vibration", "pressure", "rpm", "humidity"]


def dataset(n=1500, seed=42):
    x, y = make_classification(n_samples=n, n_features=5, n_informative=4,
        n_redundant=0, weights=[0.78, 0.22], flip_y=0.03, random_state=seed)
    return x.astype(float), y.astype(int)


def train(artifact_dir="artifacts"):
    target = Path(artifact_dir)
    target.mkdir(parents=True, exist_ok=True)
    x, y = dataset()
    # Split before fitting; evaluation set is untouched during training.
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, stratify=y, random_state=42)
    model = RandomForestClassifier(n_estimators=100, min_samples_leaf=3,
                                   random_state=42, n_jobs=1)
    model.fit(x_train, y_train)
    scores = model.predict_proba(x_test)[:, 1]
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, scores >= 0.5)), 4),
        "average_precision": round(float(average_precision_score(y_test, scores)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, scores)), 4),
        "test_rows": int(len(y_test)), "train_rows": int(len(y_train)),
        "dataset": "synthetic_make_classification", "model": "RandomForestClassifier"
    }
    joblib.dump(model, target / "model.joblib")
    np.save(target / "reference.npy", x_train)
    (target / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf8")
    return metrics


def load(artifact_dir="artifacts"):
    root=Path(artifact_dir)
    for name in ("model.joblib", "reference.npy"):
        if not (root/name).exists():
            raise FileNotFoundError("Model artifacts missing; run: python train.py")
    return joblib.load(root / "model.joblib"), np.load(root / "reference.npy")


def psi(reference, current, bins=10, epsilon=1e-6):
    """PSI using reference quantile bin boundaries and explicit +/-infinity tails.

    Numeric distribution shift heuristic, not a statistical significance test.
    """
    ref=np.asarray(reference, dtype=float)
    cur=np.asarray(current, dtype=float)
    if not len(ref) or not len(cur) or not (np.isfinite(ref).all() and np.isfinite(cur).all()):
        raise ValueError("Nonempty, finite samples required")
    if bins < 2: raise ValueError("At least two bins required")
    inner=np.unique(np.quantile(ref, np.linspace(0, 1, bins+1)[1:-1]))
    edges=np.r_[-np.inf, inner, np.inf]
    ref_counts=np.histogram(ref, bins=edges)[0].astype(float)
    cur_counts=np.histogram(cur, bins=edges)[0].astype(float)
    a=np.maximum(ref_counts / len(ref), epsilon)
    b=np.maximum(cur_counts / len(cur), epsilon)
    return float(np.sum((b-a)*np.log(b/a)))


def drift_report(reference, current, threshold=0.2):
    ref=np.asarray(reference,dtype=float)
    cur=np.asarray(current,dtype=float)
    if ref.ndim!=2 or cur.ndim!=2 or ref.shape[1]!=len(FEATURES) or cur.shape[1]!=len(FEATURES):
        raise ValueError("Expected 2D arrays with five feature columns")
    if not (0 < threshold): raise ValueError("threshold must be positive")
    features={name:round(psi(ref[:,i],cur[:,i]),6) for i,name in enumerate(FEATURES)}
    alerted=[name for name,val in features.items() if val > threshold]
    return {"rows":len(cur), "reference_rows":len(ref), "psi":features,
            "threshold":threshold,"drifted_features":alerted,"alert":bool(alerted),
            "note":"PSI is a heuristic; thresholds must be calibrated per dataset."}
