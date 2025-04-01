# A clean better logistic regression implementation

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
data = pd.read_csv('Social_Network_Ads.csv')

# Mapping the categorical variable bascially we have a variable Gender and we need to map it so that we could use it in ML
data["Gender"] = data["Gender"].map({"Male":0, "Female":1})
X = data[["Gender","Age", "EstimatedSalary"]].values
Y = data["Purchased"].values

# Feature Scaling
mean_age = np.mean(X[:, 1])
std_age = np.std(X[:, 1])
mean_salary = np.mean(X[:, 2])
std_salary = np.std(X[:, 2])

X_scaled = np.copy(X)
X_scaled[:, 1] = (X[:, 1] - mean_age) / std_age
X_scaled[:, 2] = (X[:, 2] - mean_salary) / std_salary

# User Setting
a = float(input("Enter learning rate (0.01, 0.1, 1): "))
epochs = int(input("Enter number of iterations (1000, 2000, 3000): "))

m = len(Y)
n = X_scaled.shape[1]
w = np.zeros(n)
b = 0

# Creating a function to compute the sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Creating a function to compute the logistic regression
def logistic_regression(a, epochs):
    global w,b
    for i in range(epochs):
        z = np.dot(X_scaled,w) + b
        y_hat = sigmoid(z)
        error = y_hat - Y
        dw = (1/m) * np.dot(X_scaled.T, error)
        db = (1/m) * np.sum(error)
        w -= a * dw
        b -= a * db
    print(f"Learning rate for Gender: {w[0]}, Learning rate for Age: {w[1]}, Learning rate for Salary: {w[2]}, Bias: {b}")
    return w, b

# Now we would define a function to predict the output
def predict(gender, age, salary):
    global w,b
    # Feature Scaling
    gender = 0 if gender.lower() == 'male' else 1
    age = (age - mean_age) / std_age
    salary = (salary - mean_salary) / std_salary
    # Predicting the result
    z = np.dot([gender,age, salary], w) + b
    y_hat = sigmoid(z)
    return 1 if y_hat >= 0.5 else 0

# Now we would take the user input for the age and salary
gender = input("Enter the gender (Male or Female): ")
age = float(input("Enter the age: "))
salary = float(input("Enter the salary: "))

# Main function to run the system
if age > 60 or salary > 150000 or age < 18 or salary < 15000 or gender not in ["Male", "Female"]:
    print("User is not eligible")
else:
    print("User is eligible")
    logistic_regression(a, epochs)
    result = predict(gender, age, salary)
    if result == 1:
        print("User will purchase from the adverstisement :)")
    else:
        print("User will not purchase from the adverstisement :(")
