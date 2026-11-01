import pandas as pd

normal = pd.read_parquet("DATA/FaultFree_Training.parquet")
sensors = [c for c in normal.columns if c.startswith(("xmeas", "xmv"))]
avg = normal[sensors].mean()
spread = normal[sensors].std()

faulty = pd.read_parquet("DATA/Faulty_Training.parquet",
                         filters=[("simulationRun", "<=", 50)])
after = faulty[faulty["sample"] > 20]

rows = []
for fault, g in after.groupby("faultNumber"):
    shift = ((g[sensors].mean() - avg) / spread).abs()
    rows.append({"fault": int(fault),
                 "biggest_shift": round(shift.max(), 2),
                 "sensor": shift.idxmax()})

table = pd.DataFrame(rows).sort_values("biggest_shift")
print(table.to_string(index=False))