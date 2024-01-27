from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd

# Load the data
data = pd.read_csv('ml-pillaring.csv')
data_shuffled = data.sample(frac=1, random_state=587)

# split the data to train, validation and test
train_idx = data_shuffled.index[:int(0.8 * len(data_shuffled))]
val_idx = data_shuffled.index[int(0.8 * len(data_shuffled)):int(0.9 * len(data_shuffled))]
test_idx = data_shuffled.index[int(0.9 * len(data_shuffled)):]

# split the data to train, validation and test, the first 5 columns are features, the last column is the target
X_train = data_shuffled.iloc[train_idx, :-3]
y_train = data_shuffled.iloc[train_idx, -1]

X_val = data_shuffled.iloc[val_idx, :-3]
y_val = data_shuffled.iloc[val_idx, -1]

X_test = data_shuffled.iloc[test_idx, :-3]
y_test = data_shuffled.iloc[test_idx, -1]

# Create the model
regressor = LinearRegression()
regressor.fit(X_train, y_train)
predicted = regressor.predict(X_test)
mse = mean_squared_error(y_test, predicted)
r2 = regressor.score(X_test, y_test)
print('MSE on test set: ', mse)
print('R2 on test set: ', r2)

# save the results
rf_res = pd.DataFrame({'Idx': list(test_idx), 'E': list(y_test), 'E_pred': list(predicted)})
rf_res.to_csv('./csv_data/mlr_opt_res.csv', index=False)

# save the model
import pickle

pickle.dump(regressor, open('./model_output/mlr_opt_model.pkl', 'wb'))
