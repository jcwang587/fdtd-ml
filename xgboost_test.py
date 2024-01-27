import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error

# Load the data
data = pd.read_csv('ml-pillar.csv')

# Separate features and target variable
X = data.drop(['wavelength', 'E', 'Q'], axis=1)
y = data['E']

# Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=539)

# Setting up the parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [250, 300, 325, 350, 375, 400, 450],
    'learning_rate': [0.01, 0.0125, 0.015, 0.0175, 0.02],
    'max_depth': [3, 4, 5, 6, 7],
    'colsample_bytree': [0.6, 0.65, 0.7, 0.75, 0.8],
}

# Creating the XGBRegressor model
xgb_model = XGBRegressor()

# GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='r2')
grid_search.fit(X_train, y_train)

# Best parameters and best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

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
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('XGBoost Regressor')
plt.show()
