import pandas as pd
import matplotlib.pyplot as plt

normal = pd.read_parquet("DATA/FaultFree_Training.parquet",
                         filters=[("simulationRun", "==", 1)])
faulty = pd.read_parquet("DATA/Faulty_Training.parquet",
                         filters=[("simulationRun", "==", 1)])

print("Normal rows for run 1:", len(normal))
print("Faulty rows for run 1:", len(faulty))
print("Fault numbers present:", sorted(faulty["faultNumber"].unique()))

fault1 = faulty[faulty["faultNumber"] == 1]

plt.plot(normal["sample"], normal["xmeas_1"], label="normal")
plt.plot(fault1["sample"], fault1["xmeas_1"], label="fault 1")
plt.axvline(20, color="black", linestyle="--")
plt.xlabel("reading number (1 reading = 3 minutes)")
plt.ylabel("xmeas_1 (one sensor)")
plt.legend()
plt.savefig("picture1.png")
print("Saved picture1.png")