
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

df = pd.read_csv("Rajasthan_Heatwave_2006_2025.csv")

print("Original Dataset:")
print(df.head())

# Select suitable numerical attributes
attributes = ["TEMP2M", "TMAX", "TMIN", "WIND_U10", "WIND_V10", "RAIN"]
attributes = [col for col in attributes if col in df.columns]

data = df[attributes].copy()

# Handle missing values
data = data.fillna(data.median())

# 1. Min-Max Normalization
min_max = (data - data.min()) / (data.max() - data.min()).replace(0, 1)

# 2. Z-Score Normalization
z_score = (data - data.mean()) / data.std().replace(0, 1)

# 3. Decimal Scaling Normalization
decimal = data.copy()

for col in attributes:
    maximum = data[col].abs().max()
    j = len(str(int(maximum))) if maximum >= 1 else 0
    decimal[col] = data[col] / (10 ** j)

# Display consolidated normalization output
print("\nMin-Max Normalization:")
print(min_max.head())

print("\nZ-Score Normalization:")
print(z_score.head())

print("\nDecimal Scaling Normalization:")
print(decimal.head())

# 4. K-Means Discretization
# Use normalized temperature attributes for clustering
cluster_attributes = [
    col for col in ["TEMP2M", "TMAX", "TMIN"]
    if col in min_max.columns
]

if len(cluster_attributes) >= 1:
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["Temperature_Cluster"] = kmeans.fit_predict(
        min_max[cluster_attributes]
    ) + 1

    print("\nK-Means Discretization Output:")
    print(df[cluster_attributes + ["Temperature_Cluster"]].head(10))

    print("\nNumber of records in each cluster:")
    print(df["Temperature_Cluster"].value_counts().sort_index())

# Save all results
for col in attributes:
    df["MinMax_" + col] = min_max[col]
    df["ZScore_" + col] = z_score[col]
    df["Decimal_" + col] = decimal[col]

df.to_csv("exp6_output.csv", index=False)

print("\nAll results saved in exp6_output.csv")
