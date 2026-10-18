"""Generate reproducible example API payload for testing drift."""
import json
import numpy as np
from src.core import dataset

x,_=dataset(n=1500)
rng=np.random.default_rng(123)
batch=x[:150].copy()
batch[:,0]+=3.0   # Deliberate distribution shift for demo
print(json.dumps({"records":[{"features":row.tolist()} for row in batch],"psi_threshold":0.2},indent=2))
