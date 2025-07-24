import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

sizes = np.array([800,900,1000,1100,1200])
prices = np.array([100,120,150,170,190])

model = LinearRegression()
model.fit(sizes,prices)

slope = model.coef_[0]
intercept = model.intercept_

print("Slope: ",slope)
print("Iintercept: ", intercept)

plt.scatter(sizes, prices, color='blue', label='Data')
plt.plot(sizes,model.predict(sizes), color='red', label="best fit")

plt.xlabel("House Size")
plt.ylabel("House Price")
plt.title("House Sizes vs Prices")
plt.legend()
plt.grid(True)
plt.show()


new_size = np.array([[1050]])
predicted_price = model.predict(new_size)
print(f"1100 বর্গফুটের একটি বাড়ির আনুমানিক দাম: {predicted_price[0]:.2f} হাজার ডলার")
