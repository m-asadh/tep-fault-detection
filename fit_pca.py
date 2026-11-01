import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

normal = pd.read_parquet("DATA/FaultFree_Training.parquet")
sensors = [c for c in normal.columns if c.startswith(("xmeas", "xmv"))]

avg = normal[sensors].mean()
spread = normal[sensors].std()
scaled = (normal[sensors] - avg) / spread

pca = PCA()
pca.fit(scaled)

cumulative = np.cumsum(pca.explained_variance_ratio_)
for k in [5, 10, 15, 20, 25, 30]:
    print(k, "components explain", round(cumulative[k - 1] * 100, 1), "% of normal movement")
for target in [0.90, 0.95]:
    k = int(np.argmax(cumulative >= target)) + 1
    print(target * 100, "% needs", k, "components")