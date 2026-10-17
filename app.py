"""Population Stability Index drift alert demo, with deterministic sample data."""
import json
import math
import random

def psi(reference, current, edges):
    def share(values, low, high, last=False):
        matches = sum(low <= v <= high if last else low <= v < high for v in values)
        return max(matches / len(values), 1e-6)
    score = 0.0
    for i, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        p = share(reference,low,high,i==len(edges)-2)
        q = share(current,low,high,i==len(edges)-2)
        score += (q-p)*math.log(q/p)
    return score

def main():
    random.seed(42)
    ref = [random.gauss(50,10) for _ in range(1000)]
    current = [random.gauss(59,10) for _ in range(1000)]
    edges = [-float("inf"),35,45,55,65,float("inf")]
    result = {"feature":"sensor_temperature","PSI":round(psi(ref,current,edges),4),
              "alert_threshold":0.2,"sample_type":"synthetic"}
    result["drift_alert"] = result["PSI"] > result["alert_threshold"]
    print(json.dumps(result,indent=2))

if __name__=="__main__":main()
