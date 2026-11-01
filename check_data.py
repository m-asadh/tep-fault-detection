import pandas as pd

ff_train = pd.read_parquet("DATA/FaultFree_Training.parquet")
print("Fault-free training:", ff_train.shape)

print("Missing values:", ff_train.isna().sum().sum())

sensors = [c for c in ff_train.columns if c.startswith(("xmeas", "xmv"))]
print("Number of sensor columns:", len(sensors))

print("Readings per run:", ff_train.groupby("simulationRun")["sample"].max().iloc[0])