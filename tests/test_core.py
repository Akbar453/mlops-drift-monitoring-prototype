import numpy as np
import pytest
from src.core import dataset, drift_report, psi, train, load, FEATURES

def test_psi_equal_samples():
    x=np.linspace(-2,2,200)
    assert psi(x,x)==pytest.approx(0, abs=1e-10)

def test_psi_detects_shift():
    x=np.linspace(-2,2,200)
    assert psi(x,x+4)>0.2

def test_reject_nan():
    with pytest.raises(ValueError): psi([1,float("nan")],[1,2])

def test_train_and_monitor(tmp_path):
    m=train(tmp_path)
    model, reference=load(tmp_path)
    x,_=dataset(n=100)
    report=drift_report(reference,x+np.array([4,0,0,0,0]))
    assert 0<=m["roc_auc"]<=1
    assert len(FEATURES)==5
    assert report["alert"]
    assert len(model.predict(x))==100
