import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# Example dataset: Generate synthetic data (You can replace it with any real dataset)
# Example: Predicting a target variable y based on feature x.
np.random.seed(42)
X = np.random.rand(100, 1) * 10  # Feature values (between 0 and 10)
y = 3 * X + np.random.randn(100, 1) * 2  # Target with some noise

# Convert to DataFrame (Optional for your real dataset)
data = pd.DataFrame(np.column_stack([X, y]), columns=["Feature", "Target"])

# Check the first few rows
print(data.head())
# Splitting the data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)
# Predict the target values for the test set
y_pred = model.predict(X_test)
# Evaluate using Mean Squared Error (MSE) and R-squared (R²)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
# Plot the training data and the regression line
plt.scatter(X_train, y_train, color='blue', label='Training data')
plt.plot(X_train, model.predict(X_train), color='red', label='Regression line')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Training Data and Regression Line')
plt.legend()
plt.show()
# Plot Actual vs Predicted values
plt.scatter(y_test, y_pred, color='green')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted')
plt.show()
