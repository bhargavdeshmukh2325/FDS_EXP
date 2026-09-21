import pandas as pd

df = pd.read_csv("Rajasthan_Heatwave_2006_2025.csv")

data = df["TMAX"].tolist()

start = 290
h = 5
k = 7

lower = []
upper = []
mid = []
freq = []

for i in range(k):
    lower.append(start + i * h)
    upper.append(start + (i + 1) * h)
    mid.append(start + i * h + h / 2)
    freq.append(0)

n = 0

for x in data:
    n = n + 1

    for i in range(k):

        if x >= lower[i] and x < upper[i]:
            freq[i] = freq[i] + 1
            break

cf = []
running = 0

for i in range(k):
    running = running + freq[i]
    cf.append(running)

print("Class Interval   Midpoint   Frequency   Cum Frequency")

for i in range(k):
    print(
        lower[i], "-", upper[i],
        "       ", mid[i],
        "       ", freq[i],
        "       ", cf[i]
    )

print("\nN =", n)

total = 0

for i in range(k):
    total = total + freq[i] * mid[i]

mean = total / n


def find_value(pos):

    for i in range(k):

        if cf[i] >= pos:

            before = cf[i] - freq[i]

            return lower[i] + ((pos - before) / freq[i]) * h


median = find_value(n / 2)

q1 = find_value(n / 4)

q3 = find_value(3 * n / 4)

iqr = q3 - q1


m = 0

for i in range(k):

    if freq[i] > freq[m]:
        m = i

f1 = freq[m]

if m == 0:
    f0 = 0
else:
    f0 = freq[m - 1]

if m == k - 1:
    f2 = 0
else:
    f2 = freq[m + 1]

mode = lower[m] + ((f1 - f0) / (2 * f1 - f0 - f2)) * h


ss = 0

for i in range(k):
    ss = ss + freq[i] * (mid[i] - mean) ** 2

variance = ss / (n - 1)

std = variance ** 0.5

print("\nGrouped Data Analysis")
print("Mean:", mean)
print("Median:", median)
print("Mode:", mode)
print("Variance:", variance)
print("Standard deviation:", std)
print("Q1:", q1)
print("Q3:", q3)
print("IQR:", iqr)