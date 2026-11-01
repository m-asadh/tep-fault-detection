import pandas as pd
import numpy as np
from scipy.stats import chi2
from sklearn.decomposition import PCA

normal = pd.read_parquet("DATA/FaultFree_Training.parquet")
sensors = [c for c in normal.columns if c.startswith(("xmeas", "xmv"))]
avg = normal[sensors].mean()
spread = normal[sensors].std()
scaled = (normal[sensors] - avg) / spread

K = 31
alpha = 0.01
pca = PCA()
pca.fit(scaled)
kept = pca.explained_variance_[:K]
dropped = pca.explained_variance_[K:]

t2_limit = chi2.ppf(1 - alpha, K)
theta1 = dropped.sum()
theta2 = (dropped ** 2).sum()
spe_limit = (theta2 / theta1) * chi2.ppf(1 - alpha, theta1 ** 2 / theta2)

onset = 160
rows = []
for f in range(1, 21):
    d = pd.read_parquet("DATA/Faulty_Testing.parquet",
                        filters=[("faultNumber", "==", f)])
    d = d.sort_values(["simulationRun", "sample"])
    sc = (d[sensors] - avg) / spread
    scores = pca.transform(sc)[:, :K]
    t2 = ((scores ** 2) / kept).sum(axis=1)
    rebuilt = scores @ pca.components_[:K] + pca.mean_
    spe = ((sc.values - rebuilt) ** 2).sum(axis=1)
    alarm = ((t2 > t2_limit) | (spe > spe_limit)).reshape(500, 960)
    three = alarm.copy()
    three[:, 2:] = alarm[:, 2:] & alarm[:, 1:-1] & alarm[:, :-2]
    three[:, :2] = False
    alarm = three


    before = alarm[:, :onset]
    after = alarm[:, onset:]
    detected = after.any(axis=1)
    delay = after.argmax(axis=1)[detected] + 1

    rows.append({"fault": f,
                 "false_alarm_%": round(before.mean() * 100, 1),
                 "missed_%": round((1 - after.mean()) * 100, 1),
                 "delay_readings": round(delay.mean(), 1) if detected.any() else None,
                 "runs_detected_%": round(detected.mean() * 100, 1)})
    print("done fault", f)

print(pd.DataFrame(rows).to_string(index=False))