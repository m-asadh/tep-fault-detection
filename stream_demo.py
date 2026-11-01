import pandas as pd
import numpy as np
from scipy.stats import chi2
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

W = 20
K = 31
alpha = 0.01

# ---- Part 1: learn normal behaviour (PCA) and set the alarm lines ----
normal = pd.read_parquet("DATA/FaultFree_Training.parquet")
sensors = [c for c in normal.columns if c.startswith(("xmeas", "xmv"))]
avg = normal[sensors].mean()
spread = normal[sensors].std()
scaled = (normal[sensors] - avg) / spread

pca = PCA()
pca.fit(scaled.values)
kept = pca.explained_variance_[:K]
dropped = pca.explained_variance_[K:]
t2_limit = chi2.ppf(1 - alpha, K)
theta1 = dropped.sum()
theta2 = (dropped ** 2).sum()
spe_limit = (theta2 / theta1) * chi2.ppf(1 - alpha, theta1 ** 2 / theta2)

# ---- Part 2: train the forest that names the fault ----
def load(name, start, runs):
    d = pd.read_parquet(f"DATA/{name}.parquet",
                        filters=[("simulationRun", "<=", runs)])
    d = d.sort_values(["faultNumber", "simulationRun", "sample"])
    X, y = [], []
    for (fault, run), g in d.groupby(["faultNumber", "simulationRun"]):
        x = g[sensors].values[start:]
        n = len(x) // W
        x = x[:n * W].reshape(n, W, len(sensors))
        X.append(np.hstack([x.mean(axis=1), x.std(axis=1)]))
        y.extend([int(fault)] * n)
    return np.vstack(X), np.array(y)

print("Training the forest, please wait...")
Xa, ya = load("FaultFree_Training", 0, 500)
Xb, yb = load("Faulty_Training", 20, 50)
forest = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0)
forest.fit(np.vstack([Xa, Xb]), np.concatenate([ya, yb]))
print("Forest ready.")

# ---- Part 3: replay one run, one reading at a time ----
def replay(title, name, fault, run, onset):
    flt = [("simulationRun", "==", run)]
    if fault > 0:
        flt.append(("faultNumber", "==", fault))
    d = pd.read_parquet(f"DATA/{name}.parquet", filters=flt).sort_values("sample")
    rows = d[sensors].values

    print()
    print("=====", title, "=====")
    streak = 0
    start = None
    due = None
    for i in range(1, len(rows) + 1):
        sc = (rows[i - 1] - avg.values) / spread.values
        z = pca.transform(sc.reshape(1, -1))[0, :K]
        t2 = (z ** 2 / kept).sum()
        spe = ((sc - (z @ pca.components_[:K] + pca.mean_)) ** 2).sum()
        over = (t2 > t2_limit) or (spe > spe_limit)
        streak = streak + 1 if over else 0

        if streak == 3 and due is None:
            start = i - 2
            due = start + W - 1
            if onset is None or i <= onset:
                kind = "FALSE ALARM (no fault has started)"
            else:
                kind = f"real fault caught, {i - onset} readings after it started"
            print(f"reading {i} (minute {3 * i}): ALARM confirmed - {kind}")

        if due is not None and i == due:
            window = rows[start - 1:start - 1 + W]
            feats = np.hstack([window.mean(axis=0), window.std(axis=0)]).reshape(1, -1)
            guess = int(forest.predict(feats)[0])
            if guess == 0:
                answer = "normal (the forest cannot name a fault)"
            else:
                answer = f"fault {guess}"
            print(f"reading {i} (minute {3 * i}): forest says {answer}. True answer: "
                  + ("no fault" if fault == 0 else f"fault {fault}"))
            break

    if due is None:
        print("No alarm raised in", len(rows), "readings.")

replay("Normal run", "FaultFree_Testing", 0, 7, None)
replay("Fault 1", "Faulty_Testing", 1, 7, 160)
replay("Fault 4", "Faulty_Testing", 4, 7, 160)
replay("Fault 9", "Faulty_Testing", 9, 7, 160)