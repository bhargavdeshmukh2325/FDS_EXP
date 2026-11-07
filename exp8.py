
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram

# Load Rajasthan dataset
df = pd.read_csv("Rajasthan_Heatwave_2006_2025.csv")

print("Dataset loaded successfully!")
print("Rows and columns:", df.shape)

# Convert columns into numeric values
cols = ["TMAX", "TMIN", "TEMP2M", "DEW2M",
        "RAIN", "WIND_U10", "HEATWAVE"]

for col in cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=cols + ["DISTRICT"])

# Create figure
fig = plt.figure(figsize=(16, 12))

# 1. Pixel-Oriented Visualization - Heatmap
ax1 = fig.add_subplot(2, 2, 1)

corr = df[cols].corr()

sns.heatmap(corr, annot=True, cmap="coolwarm",
            ax=ax1, fmt=".2f")

ax1.set_title("Pixel-Based Visualization - Heatmap")

# 2. Geometric Projection - 3D Scatter Plot
ax2 = fig.add_subplot(2, 2, 2, projection="3d")

sample = df.sample(min(500, len(df)), random_state=42)

ax2.scatter(sample["TEMP2M"],
            sample["DEW2M"],
            sample["TMAX"],
            c=sample["HEATWAVE"],
            cmap="coolwarm",
            marker="o")

ax2.set_title("Geometric Visualization - 3D Scatter")
ax2.set_xlabel("TEMP2M")
ax2.set_ylabel("DEW2M")
ax2.set_zlabel("TMAX")

# 3. Icon-Based Visualization
ax3 = fig.add_subplot(2, 2, 3)

normal = df[df["HEATWAVE"] == 0].sample(
    min(200, len(df[df["HEATWAVE"] == 0])),
    random_state=42
)

heat = df[df["HEATWAVE"] == 1].sample(
    min(200, len(df[df["HEATWAVE"] == 1])),
    random_state=42
)

ax3.scatter(normal["TMAX"], normal["DEW2M"],
            marker="o", label="Non-Heatwave")

ax3.scatter(heat["TMAX"], heat["DEW2M"],
            marker="^", label="Heatwave")

ax3.set_title("Icon-Based Visualization")
ax3.set_xlabel("Maximum Temperature")
ax3.set_ylabel("Dew Point")
ax3.legend()
ax3.grid()

# 4. Hierarchical Visualization - Dendrogram
ax4 = fig.add_subplot(2, 2, 4)

district_data = df.groupby("DISTRICT")[
    ["TMAX", "TEMP2M", "DEW2M"]
].mean()

scaled_data = StandardScaler().fit_transform(district_data)

linked = linkage(scaled_data, method="ward")

dendrogram(linked,
           labels=district_data.index,
           ax=ax4)

ax4.set_title("Hierarchical Visualization - Dendrogram")
ax4.set_xlabel("Districts")
ax4.set_ylabel("Distance")
ax4.tick_params(axis="x", rotation=90)

plt.suptitle("Rajasthan Heatwave Data Visualization",
             fontsize=16)

plt.tight_layout()
plt.show()

print("Experiment 8 completed successfully!")
