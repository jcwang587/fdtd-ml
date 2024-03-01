import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
import shap

# Load the data
data = pd.read_csv("ml-pillar.csv")

# Separate features and target variable
X = data.drop(["wavelength", "E", "Q"], axis=1)
y = data[["E", "Q"]]

# Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    # X_scaled, y, test_size=0.1, random_state=2546
    X_scaled, y, test_size=0.1, random_state=2737
)

# Creating the XGBRegressor model
xgb_model = XGBRegressor()

from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9],
    'learning_rate': [0.001, 0.002, 0.005, 0.01, 0.015, 0.02],
}

# Set up the grid search
grid_search = GridSearchCV(
    XGBRegressor(),
    param_grid,
    cv=3,  # Using 3-fold cross-validation
    scoring='r2',
    verbose=1,  # Shows progress
    n_jobs=-1  # Use all available cores
)

# Fit the grid search
grid_search.fit(X_train, y_train)

# Print the best parameters
print("Best parameters:", grid_search.best_params_)

# Re-train the model with the best parameters
best_params = grid_search.best_params_
optimized_xgb_model = XGBRegressor(**best_params)
optimized_xgb_model.fit(X_train, y_train)

# Predicting and evaluating the model with optimized parameters
y_pred_optimized = optimized_xgb_model.predict(X_test)

# Separate predictions for E and Q
y_pred_E = y_pred_optimized[:, 0]
y_pred_Q = y_pred_optimized[:, 1]

# Evaluate for E
r2_E = r2_score(y_test["E"], y_pred_E)
mse_E = mean_squared_error(y_test["E"], y_pred_E)

# Evaluate for Q
r2_Q = r2_score(y_test["Q"], y_pred_Q)
mse_Q = mean_squared_error(y_test["Q"], y_pred_Q)

print(f"R2 on test set for E: {r2_E:.4f}, MSE: {mse_E:.4f}")
print(f"R2 on test set for Q: {r2_Q:.4f}, MSE: {mse_Q:.4f}")

# Plotting parity plots
sns.set()
plt.figure(figsize=(16, 8))

# Parity plot for E
plt.subplot(1, 2, 1)
plt.scatter(y_test["E"], y_pred_E, s=300)
plt.plot(
    [y_test["E"].min(), y_test["E"].max()],
    [y_test["E"].min(), y_test["E"].max()],
    "k--",
)
plt.xlabel("Actual E", fontsize=32)
plt.ylabel("Predicted E", fontsize=32)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.title(f"Multi-Target XGBoost Prediction for E\n$R^2$: {r2_E:.3f}, MSE: {mse_E:.3f}", fontsize=24)

# Parity plot for Q
plt.subplot(1, 2, 2)
plt.scatter(y_test["Q"], y_pred_Q, s=300)
plt.plot(
    [y_test["Q"].min(), y_test["Q"].max()],
    [y_test["Q"].min(), y_test["Q"].max()],
    "k--",
)
plt.xlabel("Actual Q", fontsize=32)
plt.ylabel("Predicted Q", fontsize=32)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.title(f"Multi-Target XGBoost Prediction for Q\n$R^2$: {r2_Q:.3f}, MSE: {mse_Q:.3f}", fontsize=24)

plt.tight_layout()
plt.savefig("./parity_plots.svg", dpi=1200, format="svg")
plt.close()

cols = list(data.columns)[0:-3]
explainer = shap.TreeExplainer(optimized_xgb_model)
shap_values = explainer.shap_values(X_train)

shap.summary_plot(shap_values, data[cols], class_names=["E", "Q"], show=False)

fig, ax = plt.gcf(), plt.gca()
fig.set_size_inches(6, 4)

ax.set_xlabel("SHAP value", fontsize=15)
ax.spines['right'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.spines['top'].set_visible(True)
ax.spines['right'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
fig.axes[-1].yaxis.label.set_size(15)
fig.axes[-1].get_yticklabels()[0].set_fontsize(15)
fig.axes[-1].get_yticklabels()[-1].set_fontsize(15)
plt.legend(fontsize=15, frameon=False)

for label in ax.get_yticklabels():
    label.set_fontsize(15)

plt.tight_layout()
plt.savefig("./shap_summary.png", dpi=1200, format="png")
plt.close()


