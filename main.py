import pandas as pd
import numpy as np

np.random.seed(42)

n_orders = 300

data = pd.DataFrame({
    "order_id": range(n_orders),
    "processing_time": np.random.randint(5, 30, n_orders),
    "priority": np.random.randint(1, 5, n_orders),
    "zone": np.random.choice([0, 1, 2, 3], n_orders)
})

print(data.head())

# preprocessing
from sklearn.preprocessing import StandardScaler

features = data[["processing_time", "priority", "zone"]]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)


# apply k-mean
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)
data["cluster"] = kmeans.fit_predict(scaled_features)

print(data.head())

#Worker Allocation / optimisation logic
n_workers = 5

workers = {i: {"tasks": [], "total_time": 0} for i in range(n_workers)}

for _, row in data.iterrows():
    # choose worker with least work
    worker_id = min(workers, key=lambda x: workers[x]["total_time"])
    
    workers[worker_id]["tasks"].append(row["order_id"])
    workers[worker_id]["total_time"] += row["processing_time"]

# Show result
for w, info in workers.items():
    print(f"Worker {w}: Tasks={len(info['tasks'])}, Total Time={info['total_time']}")

    # Compare before and after
    # BEFORE (random assignment)
random_workers = {i: 0 for i in range(n_workers)}

for time in data["processing_time"]:
    w = np.random.randint(0, n_workers)
    random_workers[w] += time

print("\nBefore Optimisation:", random_workers)

# AFTER
optimised = {w: int(info["total_time"]) for w, info in workers.items()}
print("After Optimisation:", optimised)


# Graph
import matplotlib.pyplot as plt

plt.bar(optimised.keys(), optimised.values())
plt.title("Workload per Worker (Optimised)")
plt.xlabel("Worker")
plt.ylabel("Total Processing Time")
plt.show()