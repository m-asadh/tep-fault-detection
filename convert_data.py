import pyreadr

names = ["FaultFree_Training", "FaultFree_Testing",
         "Faulty_Training", "Faulty_Testing"]

for name in names:
    res = pyreadr.read_r(f"DATA/TEP_{name}.RData")
    df = list(res.values())[0]
    df.to_parquet(f"DATA/{name}.parquet")
    print(name, df.shape)
    del df, res