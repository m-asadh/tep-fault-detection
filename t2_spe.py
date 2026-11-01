import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

normal = pd.read_parquet("DATA/FaultFree_Training.parquet")
sensors = [c for c in normal.columns if c.startswith(("xmeas", "xmv"))]

avg = normal[sensors].mean()
spread = normal[sensors].std()
scaled = (normal[sensors] - avg) / spread

K = 31
pca = PCA(n_components=K)
pca.fit(scaled)

scores = pca.transform(scaled)
t2 = ((scores ** 2) / pca.explained_variance_).sum(axis=1)

rebuilt = pca.inverse_transform(scores)
spe = ((scaled.values - rebuilt) ** 2).sum(axis=1)

print("Average T2 on normal data:", round(t2.mean(), 2))
print("Average SPE on normal data:", round(spe.mean(), 2))
print("Expected T2 average: about", K)
print("Expected SPE average: about", round(52 * (1 - pca.explained_variance_ratio_.sum()), 2))