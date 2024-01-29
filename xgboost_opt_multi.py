import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor

# Load the data
data = pd.read_csv('ml-pillar.csv')

# Separate features and target variable
X = data.drop(['wavelength', 'E', 'Q'], axis=1)
y = data[['E', 'Q']]  # Multi-target

# Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=2737)

# Creating and training the XGBRegressor model
best_params = {'colsample_bytree': 0.7, 'learning_rate': 0.015, 'max_depth': 5, 'n_estimators': 350}
best_xgb_model = XGBRegressor(**best_params)
best_xgb_model.fit(X_train, y_train)

# Predicting
y_pred = best_xgb_model.predict(X_test)

# Separate predictions for E and Q
y_pred_E = y_pred[:, 0]
y_pred_Q = y_pred[:, 1]

# Evaluate for E
r2_E = r2_score(y_test['E'], y_pred_E)
mse_E = mean_squared_error(y_test['E'], y_pred_E)

# Evaluate for Q
r2_Q = r2_score(y_test['Q'], y_pred_Q)
mse_Q = mean_squared_error(y_test['Q'], y_pred_Q)

# Plotting parity plots
sns.set()
plt.figure(figsize=(16, 8))

# Parity plot for E
plt.subplot(1, 2, 1)
plt.scatter(y_test['E'], y_pred_E, alpha=0.7)
plt.plot([y_test['E'].min(), y_test['E'].max()], [y_test['E'].min(), y_test['E'].max()], 'k--')
plt.xlabel('Actual E')
plt.ylabel('Predicted E')
plt.title(f'Parity Plot for E\nR2: {r2_E:.2f}, MSE: {mse_E:.2f}')

# Parity plot for Q
plt.subplot(1, 2, 2)
plt.scatter(y_test['Q'], y_pred_Q, alpha=0.7)
plt.plot([y_test['Q'].min(), y_test['Q'].max()], [y_test['Q'].min(), y_test['Q'].max()], 'k--')
plt.xlabel('Actual Q')
plt.ylabel('Predicted Q')
plt.title(f'Parity Plot for Q\nR2: {r2_Q:.2f}, MSE: {mse_Q:.2f}')

plt.tight_layout()
plt.show()
