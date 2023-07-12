from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import data_set_optimization as dso
import data_exploration as de
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer

# Get Data
data = de.read_data("House_Rent_Dataset.csv")
data_with_normal_deviation = dso.get_data_with_normal_deviation(data, "Rent")

# Get features and target
X_normal_deviation, Y_normal_deviation = de.split_features_and_target(data_with_normal_deviation, "Rent")

# Get train, eval and test sets
X_train, X_eval, X_test, Y_train, Y_eval, Y_test = de.split_train_eval_test(X_normal_deviation,
                                                                            Y_normal_deviation,
                                                                            test_size=0.15,
                                                                            eval_size=0.15)

# Configure Pre-Processors
numeric_transformer = SimpleImputer(strategy='mean')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessors = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, dso.only_numeric_column_names(X_normal_deviation)),
        ("categorical", categorical_transformer, dso.only_non_numeric_column_names(X_normal_deviation))
    ]
)


pipeline = Pipeline(steps=[
    ('preprocessor', preprocessors),
    ('model', XGBRegressor(n_estimators=500, learning_rate=0.05, n_jobs=10))
])


pipeline.fit(X_train, Y_train)

scores = cross_val_score(pipeline, X_test, Y_test, cv=5, scoring='neg_mean_absolute_error')
print("MAE: %0.2f (+/- %0.2f)" % (scores.mean() * (-1), scores.std() * 2))


print("Prediction mae: {}".format(mean_absolute_error(pipeline.predict(X_test), Y_test)))