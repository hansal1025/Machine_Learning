# Create a linear regression model using real life data

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import data into the system
data = pd.read_csv('coffee_shop_revenue.csv')
X = (data[["Number_of_Customers_Per_Day"]].values)
Y = data["Daily_Revenue"].values

# Feature Scaling
mean_customers = np.mean(X)
std_customers = np.std(X)

X_scaled = np.copy(X)
X_scaled = (X - mean_customers) / std_customers

# User Setting
a = float(input("Enter learning rate (0.01, 0.1, 1): "))
epochs = int(input("Enter number of iterations (1000, 2000, 3000): "))

m = len(X_scaled)  # Number of samples
n = 1  # Number of features
w = np.zeros(n)
b = 0

# Defining the function to compute the linear regression
def linear_regression(a, epochs):
    global w,b
    for i in range(epochs):
        y_hat = np.dot(X_scaled, w) + b
        error = y_hat - Y
        dw = np.dot(X_scaled.T,error) * (1/m)
        db = np.sum(error) * (1/m)
        w -= a * dw
        b -= a * db
    print(f"Learning rate for Number of Customers: {w[0]}, Bias: {b}")
    return w, b

# We would define a function to predict the output price for out coffee shop
def predict_price(customer_daily):
    global w,b
    # Feature Scaling
    customer_daily = (customer_daily - mean_customers) / std_customers
    # Predicting the result
    y_hat = np.dot(customer_daily, w) + b
    return y_hat

# Take the user input for the number of customers
customer_daily = float(input("Enter the number of customers per day: "))

# The main function that would call the other functions
linear_regression(a, epochs)
predict_price(customer_daily)
print(f"Predicted daily revenue for {customer_daily} customers: {predict_price(customer_daily)}")

# Plotting the data
plt.scatter(X, Y, color='blue', label="Actual Data")
plt.plot(X, np.dot(X_scaled, w) + b, color='red', label="Predicted Line")
plt.title('Daily Revenue vs Number of Customers')
plt.xlabel('Number of Customers')
plt.ylabel('Daily Revenue')
plt.legend()
plt.show()