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

test = pd.read_parquet("DATA/FaultFree_Testing.parquet")
test = test.sort_values(["simulationRun", "sample"])
sc = (test[sensors] - avg) / spread
scores = pca.transform(sc)[:, :K]
t2 = ((scores ** 2) / kept).sum(axis=1)
rebuilt = scores @ pca.components_[:K] + pca.mean_
spe = ((sc.values - rebuilt) ** 2).sum(axis=1)

alarm = ((t2 > t2_limit) | (spe > spe_limit)).reshape(500, 960)
three = alarm.copy()
three[:, 2:] = alarm[:, 2:] & alarm[:, 1:-1] & alarm[:, :-2]
three[:, :2] = False

after = three[:, 160:]
print("Share of normal readings flagged:", round(after.mean() * 100, 3), "%")
print("Share of normal runs with at least one alarm:", round(after.any(axis=1).mean() * 100, 1), "%")