import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

W = 20
RUNS = 50
NORMAL_RUNS = 500

def load(name, start, runs):
    d = pd.read_parquet(f"DATA/{name}.parquet",
                        filters=[("simulationRun", "<=", runs)])
    sensors = [c for c in d.columns if c.startswith(("xmeas", "xmv"))]
    d = d.sort_values(["faultNumber", "simulationRun", "sample"])
    X, y = [], []
    for (fault, run), g in d.groupby(["faultNumber", "simulationRun"]):
        x = g[sensors].values[start:]
        n = len(x) // W
        x = x[:n * W].reshape(n, W, len(sensors))
        X.append(np.hstack([x.mean(axis=1), x.std(axis=1)]))
        y.extend([int(fault)] * n)
    return np.vstack(X), np.array(y)

Xa, ya = load("FaultFree_Training", 0, NORMAL_RUNS)
Xb, yb = load("Faulty_Training", 20, RUNS)
Xtrain = np.vstack([Xa, Xb])
ytrain = np.concatenate([ya, yb])
print("Training windows:", Xtrain.shape)

Xc, yc = load("FaultFree_Testing", 160, RUNS)
Xd, yd = load("Faulty_Testing", 160, RUNS)
Xtest = np.vstack([Xc, Xd])
ytest = np.concatenate([yc, yd])
print("Testing windows:", Xtest.shape)

forest = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0)
forest.fit(Xtrain, ytrain)
pred = forest.predict(Xtest)
print("Overall accuracy:", round(accuracy_score(ytest, pred) * 100, 1), "%")

is_fault = ytest > 0
print("Normal windows called normal:", round((pred[~is_fault] == 0).mean() * 100, 1), "%")
print("Faulty windows called some fault:", round((pred[is_fault] > 0).mean() * 100, 1), "%")
other17 = ~np.isin(ytest, [0, 3, 9, 15])
print("Accuracy on the other 17 faults:", round((pred[other17] == ytest[other17]).mean() * 100, 1), "%")

labels = list(range(21))
cm = confusion_matrix(ytest, pred, labels=labels)
recall = cm.diagonal() / cm.sum(axis=1)
wrong = cm.copy()
np.fill_diagonal(wrong, 0)
table = pd.DataFrame({"class": labels,
                      "correct_%": (recall * 100).round(1),
                      "most_confused_with": wrong.argmax(axis=1)})
print(table.to_string(index=False))

block = [0, 3, 9, 15]
print("Look-alike block (rows = truth, columns = forest's answer):")
print(pd.DataFrame(cm[np.ix_(block, block)], index=block, columns=block))