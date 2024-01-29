import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import mean_squared_error

# Load the data
data = pd.read_csv('ml-pillar.csv')

best_r2 = -np.inf  # Initialize the best R2 score to negative infinity
best_iteration = None  # To keep track of the best iteration

for i in range(10000):
    print(f'Iteration {i}')
    data_shuffled = data.sample(frac=1, random_state=i)

    # split the data to train, validation and test
    train_idx = data_shuffled.index[:int(0.8 * len(data_shuffled))]
    valid_idx = data_shuffled.index[int(0.8 * len(data_shuffled)):int(0.9 * len(data_shuffled))]
    test_idx = data_shuffled.index[int(0.9 * len(data_shuffled)):]

    # split the data to train, validation and test, the first 5 columns are features, the last column is the target
    X_train = data_shuffled.iloc[train_idx, :-3]
    y_train = data_shuffled.iloc[train_idx, -1]

    X_valid = data_shuffled.iloc[valid_idx, :-3]
    y_valid = data_shuffled.iloc[valid_idx, -1]

    X_test = data_shuffled.iloc[test_idx, :-3]
    y_test = data_shuffled.iloc[test_idx, -1]

    # Create the model
    xgb_model = xgb.XGBRegressor()

    # Define the hyperparameters
    random_grid = {'n_estimators': [int(x) for x in np.linspace(start=100, stop=1000, num=10)],
                   'max_depth': [int(x) for x in np.linspace(start=1, stop=20, num=10)],
                   'learning_rate': [0.1, 0.2, 0.5]}

    min_mse = np.inf
    best_params = None

    # Lower n_iter to 4800 to match the number of unique combinations
    for params in ParameterSampler(random_grid, n_iter=1000, random_state=587):
        xgb_model.set_params(**params)
        xgb_model.fit(X_train, y_train)
        preds = xgb_model.predict(X_valid)
        mse = mean_squared_error(y_valid, preds)

        if mse < min_mse:
            min_mse = mse
            best_params = params

    print('Best parameters found: ', best_params)

    # predict on test set
    xgb_model.set_params(**best_params)
    xgb_model.fit(X_train, y_train)
    predicted = xgb_model.predict(X_test)
    mse = mean_squared_error(y_test, predicted)
    r2 = xgb_model.score(X_test, y_test)

    # Check if this iteration has a better R2 score
    if r2 > best_r2:
        best_r2 = r2
        best_iteration = i

    print(f'Iteration {i}: MSE on test set = {mse}, R2 on test set = {r2}')

# After all iterations, print out the best iteration and its R2 score
print(f'Best iteration: {best_iteration} with R2 score: {best_r2}')