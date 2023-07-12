import data_set_optimization as dso
import data_exploration as de
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor

# Get Data
data = de.read_data("House_Rent_Dataset.csv")
data_with_normal_deviation = dso.get_data_with_normal_deviation(data, "Rent")

# Get features and target
X,Y = de.split_features_and_target(data, "Rent")
X_normal_deviation, Y_normal_deviation = de.split_features_and_target(data_with_normal_deviation, "Rent")

# Get train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
normal_deviation_X_train, normal_deviation_X_test, normal_deviation_Y_train, normal_deviation_Y_test = train_test_split(X_normal_deviation, Y_normal_deviation, test_size=0.2, random_state=0)

# Encode the data
encoded_X_train, encoded_X_test = dso.apply_ordinal_encoding(X_train, X_test)
one_hot_encoded_X_train, one_hot_encoded_X_test = dso.apply_one_hot_encoding(X_train, X_test)
one_hot_encoded_normal_deviation_X_train, one_hot_encoded_normal_deviation_X_test = dso.apply_one_hot_encoding(normal_deviation_X_train, normal_deviation_X_test)


# Train the models
encoded_trained_model = RandomForestRegressor()
encoded_trained_model.fit(encoded_X_train, Y_train)
one_hot_encoded_model = RandomForestRegressor()
one_hot_encoded_model.fit(one_hot_encoded_X_train, Y_train)
normal_deviation_one_hot_encoded_model = RandomForestRegressor()
normal_deviation_one_hot_encoded_model.fit(one_hot_encoded_normal_deviation_X_train, normal_deviation_Y_train)
xgboost_normal_deviation_one_hot_encoded_model = XGBRegressor(n_estimators=1000, learning_rate=0.1, n_jobs=6, early_stopping_rounds=5)
xgboost_normal_deviation_one_hot_encoded_model.fit(one_hot_encoded_normal_deviation_X_train, normal_deviation_Y_train,
                                                   eval_set=[(one_hot_encoded_normal_deviation_X_test, normal_deviation_Y_test)],
                                                   verbose=False)


# Evaluate the models
print("Encoded: {}".format(mean_absolute_error(Y_test, encoded_trained_model.predict(encoded_X_test))))
print("One hot encoded: {}".format(mean_absolute_error(Y_test, one_hot_encoded_model.predict(one_hot_encoded_X_test))))
print("Normal deviation one hot encoded: {}".format(mean_absolute_error(normal_deviation_Y_test, normal_deviation_one_hot_encoded_model.predict(one_hot_encoded_normal_deviation_X_test))))
print("XGBoost with normal deviation one hot encoded: {}".format(mean_absolute_error(normal_deviation_Y_test, xgboost_normal_deviation_one_hot_encoded_model.predict(one_hot_encoded_normal_deviation_X_test))))
