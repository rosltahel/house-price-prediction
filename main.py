import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
df = pd.read_csv('AmesHousing.csv')

# Select relevant features (expanded list)
features = [
    'Lot Area', 'Year Built', 'Gr Liv Area', 'Bedroom AbvGr',
    'Overall Qual', 'Garage Area', 'Total Bsmt SF', '1st Flr SF', 
    'Full Bath', 'Fireplaces'
]
X = df[features].copy()  # Create a copy of the features DataFrame
y = df['SalePrice']  # Target variable

# Handle missing values by filling them with the mean
X.fillna(X.mean(), inplace=True)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_y_pred = lr_model.predict(X_test)

# Evaluate Linear Regression
lr_mse = mean_squared_error(y_test, lr_y_pred)
lr_r2 = r2_score(y_test, lr_y_pred)

print("Linear Regression:")
print(f"Mean Squared Error: {lr_mse}")
print(f"R² Score: {lr_r2}")

# Random Forest Model
rf_model = RandomForestRegressor(random_state=42, n_estimators=100)
rf_model.fit(X_train, y_train)
rf_y_pred = rf_model.predict(X_test)

# Evaluate Random Forest
rf_mse = mean_squared_error(y_test, rf_y_pred)
rf_r2 = r2_score(y_test, rf_y_pred)

print("\nRandom Forest:")
print(f"Mean Squared Error: {rf_mse}")
print(f"R² Score: {rf_r2}")

# Visualize predictions vs actual using bar plots
num_samples = 20  # Number of samples to display
indices = np.arange(num_samples)

plt.figure(figsize=(14, 7))

# Linear Regression Bar Plot
plt.subplot(1, 2, 1)
plt.bar(indices - 0.2, y_test.iloc[:num_samples], width=0.4, label='Actual', alpha=0.7)
plt.bar(indices + 0.2, lr_y_pred[:num_samples], width=0.4, label='Predicted', alpha=0.7)
plt.xlabel("Sample Index")
plt.ylabel("House Price")
plt.title("Linear Regression: Actual vs Predicted")
plt.legend()

# Random Forest Bar Plot
plt.subplot(1, 2, 2)
plt.bar(indices - 0.2, y_test.iloc[:num_samples], width=0.4, label='Actual', alpha=0.7)
plt.bar(indices + 0.2, rf_y_pred[:num_samples], width=0.4, label='Predicted', alpha=0.7, color='orange')
plt.xlabel("Sample Index")
plt.ylabel("House Price")
plt.title("Random Forest: Actual vs Predicted")
plt.legend()

plt.tight_layout()
plt.show()
