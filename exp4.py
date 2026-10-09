
import pandas as pd
import math
import matplotlib.pyplot as plt

df = pd.read_csv("heatwave_dataset.csv")

def correlation(x, y):
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n

    num = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    dx = sum((i - mx) ** 2 for i in x)
    dy = sum((i - my) ** 2 for i in y)

    den = math.sqrt(dx * dy)

    if den == 0:
        return 0

    return num / den


data = df[["TEMP2M", "TMAX"]].dropna()

x1 = data["TEMP2M"].tolist()
y1 = data["TMAX"].tolist()

print("Positive/Actual Dataset Correlation:",
      correlation(x1, y1))

plt.scatter(x1, y1)
plt.xlabel("TEMP2M")
plt.ylabel("TMAX")
plt.title("Rajasthan: TEMP2M vs TMAX")
plt.show()

x2 = list(range(len(data)))
y2 = sorted(data["TMAX"].tolist(), reverse=True)

print("Negative Correlation:", correlation(x2, y2))

plt.scatter(x2, y2)
plt.xlabel("Row Number")
plt.ylabel("TMAX")
plt.title("Negative Correlation Example")
plt.show()

x3 = [-2, -1, 0, 1, 2] * (len(data) // 5)
x3 += [-2, -1, 0, 1, 2][:len(data) % 5]

y3 = [v ** 2 for v in x3]

print("Zero Correlation:", correlation(x3, y3))

plt.scatter(x3, y3)
plt.xlabel("X values")
plt.ylabel("Y values")
plt.title("Zero Correlation Example")
plt.show()