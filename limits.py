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
g = theta2 / theta1
h = theta1 ** 2 / theta2
spe_limit = g * chi2.ppf(1 - alpha, h)

print("T2 limit:", round(t2_limit, 2))
print("SPE limit:", round(spe_limit, 2))

test = pd.read_parquet("DATA/FaultFree_Testing.parquet")
test_scaled = (test[sensors] - avg) / spread
scores = pca.transform(test_scaled)[:, :K]
t2 = ((scores ** 2) / kept).sum(axis=1)
rebuilt = scores @ pca.components_[:K] + pca.mean_
spe = ((test_scaled.values - rebuilt) ** 2).sum(axis=1)

print("False alarms, T2 alone:", round((t2 > t2_limit).mean() * 100, 2), "%")
print("False alarms, SPE alone:", round((spe > spe_limit).mean() * 100, 2), "%")
print("False alarms, either:", round(((t2 > t2_limit) | (spe > spe_limit)).mean() * 100, 2), "%")