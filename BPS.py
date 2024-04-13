import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import csv
import random
from datetime import datetime, timedelta

# Function to generate synthetic sales data
def generate_sales_data(start_date, end_date, num_records):
    data = []
    current_date = start_date
    for _ in range(num_records):
        sales_quantity = random.randint(50, 200)  # Random sales quantity
        data.append([current_date.strftime('%Y-%m-%d'), sales_quantity])
        current_date += timedelta(days=1)
    return data

# Function to simulate production process and detect anomalies
def simulate_production_process():
    # Simulate production process and detect anomalies
    # Example implementation goes here
    pass

# Function to analyze quality control data and implement corrective actions
def analyze_quality_control_data():
    # Analyze quality control data and implement corrective actions
    # Example implementation goes here
    pass

# Function to integrate with suppliers and distributors for supply chain collaboration
def supply_chain_integration():
    # Integrate with suppliers and distributors for supply chain collaboration
    # Example implementation goes here
    pass

# Define start and end dates for the data
start_date = datetime(2020, 1, 1)
end_date = datetime(2020, 12, 31)

# Number of records to generate
num_records = (end_date - start_date).days + 1

# Generate synthetic sales data
sales_data = generate_sales_data(start_date, end_date, num_records)

# Write sales data to CSV file
sales_csv_file = 'sales_data.csv'
with open(sales_csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['date', 'sales_quantity'])  # Write header
    writer.writerows(sales_data)

print(f"CSV file '{sales_csv_file}' created successfully.")

# Load historical sales data
sales_data = pd.read_csv(sales_csv_file)

# Data Cleaning and Preprocessing
# Convert 'date' column to datetime type
sales_data['date'] = pd.to_datetime(sales_data['date'])

# Check for missing values
missing_values = sales_data.isnull().sum()
print("Missing values:\n", missing_values)

# Handling Outliers
# No outliers handling in this example

# Handling Duplicate Entries
sales_data.drop_duplicates(inplace=True)

# Feature Engineering
# Extracting date features
sales_data['year'] = sales_data['date'].dt.year
sales_data['month'] = sales_data['date'].dt.month
sales_data['day'] = sales_data['date'].dt.day

# Select features and target variable for demand forecasting
demand_features = sales_data[['year', 'month', 'day']]
demand_target = sales_data['sales_quantity']

# Split data into training and testing sets for demand forecasting
X_train_demand, X_test_demand, y_train_demand, y_test_demand = train_test_split(demand_features, demand_target, test_size=0.2, random_state=42)

# Train Random Forest model for demand forecasting
rf_demand_model = RandomForestRegressor()
rf_demand_model.fit(X_train_demand, y_train_demand)

# Predict demand
demand_predictions = rf_demand_model.predict(X_test_demand)

# Evaluate demand forecasting model performance
demand_mse = mean_squared_error(y_test_demand, demand_predictions)
print("Demand Forecasting Mean Squared Error:", demand_mse)

# Visualize actual vs. predicted demand
plt.figure(figsize=(10, 6))
plt.plot(y_test_demand.values, label='Actual')
plt.plot(demand_predictions, label='Predicted')
plt.title('Actual vs. Predicted Demand')
plt.xlabel('Sample')
plt.ylabel('Demand')
plt.legend()
plt.show()

# Simulate production process and detect anomalies
simulate_production_process()

# Analyze quality control data and implement corrective actions
analyze_quality_control_data()

# Integrate with suppliers and distributors for supply chain collaboration
supply_chain_integration()
