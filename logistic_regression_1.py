#We would create a model training alogorithm using logistic regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
data = pd.read_csv("Social_Network_Ads.csv")
# print(data.head())
X = data[["Age","EstimatedSalary"]].values
Y = data["Purchased"].values

# All the defining features
m = len(Y)

# # Convert the Y data into Scale and the problem is in this type of scaling
# max_salary = data["EstimatedSalary"].max()
# max_age = data["Age"].max()
# X_scaled = X.copy().astype(np.float64) # Create a copy of X to avoid modifying the original data
# X_scaled[:,1] = X[:,1] / max_salary
# X_scaled[:,0] = X[:,0] / max_age

#We must use StandardScaler to scale the data which is a better approach where (x-u) / s where u is the mean and s is the standard deviation



# All the defining features
m = len(Y)
n = X_scaled.shape[1]
w = np.zeros(n)  # weights
b = 0   # bias
iterations = 1000 # number of iterations

def logistic_regression(a, epochs=1000):
    global w,b # We want to update global weights and bias
    for _ in range(epochs):
        z = np.dot(X_scaled,w) + b  # linear combination
        y_hat = 1 / (1+np.exp(-z)) # sigmoid function
        error = y_hat - Y
        # w = w - a * error * X_scaled[i] # gradient descent
        # b = b - a * error # update bias
        dw = np.dot(X_scaled.T,error) / m # gradient descent for weights
        db = np.sum(error) / m # gradient descent for bias
        w -= a*dw # update weights
        b -= a*db   # update bias
    print(f"Learning Rate for Age: {w[0]}, Learning Rate for Salary: {w[1]}, Bias: {b}")

def predict(age, salary):
    age_scaled = age / max_age
    salary_scaled = salary / max_salary
    z = w[0] * age_scaled + w[1] * salary_scaled + b
    y_hat = 1 / (1 + np.exp(-z))
    print(f"Predicted value: {y_hat}")
    return 1 if (y_hat >=0.5) else 0

a = float(input("Input the learning rate: "))
age = int(input("Input the age of the customer (below 60): "))
salary = int(input("Input the salary of the customer (below 150000): "))

if age > 60 or salary > 150000:
    print("Age and Salary should be below 60 and 150000 respectively.")
else:
    print(f"Age: {age}, Salary: {salary}")
    print("Training the model...")

#-------------------------------------------
logistic_regression(a)
purchase = predict(age, salary)
print(purchase)
if purchase == 1:
    print("The user will buy it! :)")
else:
    print("The user won't buy it! :(")
