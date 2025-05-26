import numpy as nm
import matplotlib.pyplot as plt

data = [1, 10, 11, 12, 21, 34, 31, 33, 40, 45, 100]

variance = nm.var(data)
print(f"Variance of data: {variance:.2f}")

plt.figure(figsize=(10, 5))
plt.plot(data, marker='o', label='Data Points')
plt.axhline(y=nm.mean(data), color='red', linestyle='--', label='Mean')
plt.title(f"Data Plot with Variance = {variance:.2f}")
plt.xlabel('Index')
plt.ylabel('Value')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
