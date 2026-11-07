
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load Rajasthan dataset
df = pd.read_csv("Rajasthan_Heatwave_2006_2025.csv")

print("Dataset loaded successfully!")
print("Rows and columns:", df.shape)

# Select required columns
df["TMAX"] = pd.to_numeric(df["TMAX"], errors="coerce")
df["TEMP2M"] = pd.to_numeric(df["TEMP2M"], errors="coerce")
df["HEATWAVE"] = pd.to_numeric(df["HEATWAVE"], errors="coerce")

df = df.dropna(subset=["TMAX", "TEMP2M", "HEATWAVE"])

# Sort dataset
df = df.sort_values(["YEAR", "MONTH", "DAY"])

# Create figure
fig, ax = plt.subplots(2, 3, figsize=(15, 10))

# 1. Line Plot
ax[0, 0].plot(df["TMAX"].head(50), marker="o")
ax[0, 0].set_title("Line Plot - Maximum Temperature")
ax[0, 0].set_xlabel("Sample Index")
ax[0, 0].set_ylabel("TMAX")
ax[0, 0].grid(True)

# 2. Bar Plot
monthly = df.groupby("MONTH")["TMAX"].mean()

ax[0, 1].bar(monthly.index, monthly.values)
ax[0, 1].set_title("Bar Plot - Monthly Average Temperature")
ax[0, 1].set_xlabel("Month")
ax[0, 1].set_ylabel("Average TMAX")
ax[0, 1].grid(axis="y")

# 3. Scatter Plot
ax[0, 2].scatter(df["TEMP2M"].head(200),
                 df["TMAX"].head(200))

ax[0, 2].set_title("Scatter Plot - TEMP2M vs TMAX")
ax[0, 2].set_xlabel("TEMP2M")
ax[0, 2].set_ylabel("TMAX")
ax[0, 2].grid(True)

# 4. Pie Plot
heatwave = df["HEATWAVE"].value_counts().sort_index()

ax[1, 0].pie(
    heatwave,
    labels=heatwave.index,
    autopct="%1.1f%%",
    startangle=140
)

ax[1, 0].set_title("Pie Plot - Heatwave Distribution")

# 5. Box Plot
ax[1, 1].boxplot(df["TMAX"])
ax[1, 1].set_title("Box Plot - Maximum Temperature")
ax[1, 1].set_ylabel("TMAX")

# 6. Histogram
ax[1, 2].hist(df["TMAX"], bins=20, edgecolor="black")
ax[1, 2].set_title("Histogram - Temperature Distribution")
ax[1, 2].set_xlabel("Maximum Temperature")
ax[1, 2].set_ylabel("Frequency")
ax[1, 2].grid(True)

# Display all graphs
plt.suptitle("Rajasthan Heatwave Dataset Visualization",
             fontsize=15)

plt.tight_layout()
plt.show()
