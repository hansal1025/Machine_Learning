# Create a linear regression model using real life data

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import data into the system
data = pd.read_csv('coffee_shop_revenue.csv')
X = data[["Number_of_Customers_Per_Day",
          "Average_Order_Value",
          "Operating_Hours_Per_Day",
          "Number_of_Employees",
          "Marketing_Spend_Per_Day",
          "Location_Foot_Traffic"]].values
Y = data["Daily_Revenue"].values

# Feature Scaling
mean_customers = np.mean(X[:,0])
std_customers = np.std(X[:,0])
mean_order_value = np.mean(X[:,1])
std_order_value = np.std(X[:,1])
mean_hours = np.mean(X[:,2])
std_hours = np.std(X[:,2])
mean_employees = np.mean(X[:,3])
std_employees = np.std(X[:,3])
mean_marketing = np.mean(X[:,4])
std_marketing = np.std(X[:,4])
mean_traffic = np.mean(X[:,5])
std_traffic = np.std(X[:,5])

X_scaled = np.copy(X)
X_scaled[:,0] = (X[:,0] - mean_customers) / std_customers
X_scaled[:,1] = (X[:,1] - mean_order_value) / std_order_value
X_scaled[:,2] = (X[:,2] - mean_hours) / std_hours
X_scaled[:,3] = (X[:,3] - mean_employees) / std_employees
X_scaled[:,4] = (X[:,4] - mean_marketing) / std_marketing
X_scaled[:,5] = (X[:,5] - mean_traffic) / std_traffic

# User Setting
a = float(input("Enter learning rate (0.01, 0.1, 1): "))
epochs = int(input("Enter number of iterations (1000, 2000, 3000): "))

m = len(X_scaled)  # Number of samples
n = 6  # Number of features
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
    print(f"Learning rate for Number of Customers: {w[0]} - {w[1]} - {w[2]} - {w[3]} - {w[4]} - {w[5]}, Bias: {b}")
    return w, b

# We would define a function to predict the output price for out coffee shop
def predict_price(customer_daily,avg_value,operating_hours, no_employees, marketing_spend, foot_traffic):
    global w,b
    # Feature Scaling
    customer_daily = (customer_daily - mean_customers) / std_customers
    avg_value = (avg_value - mean_order_value) / std_order_value
    operating_hours = (operating_hours - mean_hours) / std_hours
    no_employees = (no_employees - mean_employees) / std_employees
    marketing_spend = (marketing_spend - mean_marketing) / std_marketing
    foot_traffic = (foot_traffic - mean_traffic) / std_traffic
    # Predicting the result
    y_hat = w[0] * customer_daily + w[1] * avg_value + w[2] * operating_hours + w[3] * no_employees + w[4] * marketing_spend + w[5] * foot_traffic + b
    return y_hat

# Take the user input for the number of customers
customer_daily = float(input("Enter the number of customers per day: "))
avg_value = float(input("Enter the average order value: "))
operating_hours = int(input("Enter the operating hours per day: "))
no_employees = int(input("Enter the number of employees: "))
marketing_spend = float(input("Enter the marketing spend per day: "))
foot_traffic = float(input("Enter the location foot traffic: "))

# The main function that would call the other functions
linear_regression(a, epochs)
prediction = predict_price(customer_daily,avg_value,operating_hours, no_employees, marketing_spend, foot_traffic )
print(f"Predicted daily revenue for {customer_daily} customers, {avg_value} average value, {operating_hours} hrs, {no_employees} employees, {marketing_spend} spend, {foot_traffic} traffic: {prediction}")

# # Plotting the data - only when there is 1 parameter
# plt.scatter(X[:,0], Y, color='blue', label="Actual Data")
# plt.plot(X[:,0], np.dot(X_scaled, w) + b, color='red', label="Predicted Line")
# plt.title('Daily Revenue vs Number of Customers')
# plt.xlabel('Number of Customers')
# plt.ylabel('Daily Revenue')
# plt.legend()
# plt.show()