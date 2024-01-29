import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Load the data
data = pd.read_csv("ml-pillar.csv")

# Separate features and target variable
X = data.drop(["wavelength", "E", "Q"], axis=1)
y = data["Q"]

# Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.1, random_state=1204
)

# Creating the XGBRegressor model
xgb_model = XGBRegressor()

# Set the best parameters
best_params = {
    "colsample_bytree": 0.7,
    "learning_rate": 0.015,
    "max_depth": 5,
    "n_estimators": 350,
}

# Training the model with the best parameters
best_xgb_model = XGBRegressor(**best_params)
best_xgb_model.fit(X_train, y_train)

# Predicting and evaluating the model
y_pred = best_xgb_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# Plot the parity plot
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], "k--")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("XGBoost Regressor")
plt.show()
