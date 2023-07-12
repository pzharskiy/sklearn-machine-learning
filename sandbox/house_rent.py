from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

import intermediate_machine_learning.data_exploration as de
import intermediate_machine_learning.data_set_optimization as dso

# I use this hack: I implemented custom XGBoostWithEarlyStop class, and inside of fit() I parse X into train and eval.
# This was done for using early_stopping_rounds, early_stopping_rounds doesnt work without provided eval_set
# In this implementation, when grid(fit) I send my data thought the transformers
# and then to estinator XGBoostRegressorWithEarlyStop(), where I split data to train and validation set

class XGBoostWithEarlyStop(BaseEstimator):

    def __init__(self,
                 #early_stopping_rounds=5,
                 eval_size=0.1,
                 eval_metric='mae',
                 verbose=False,  # Verbose for printing, if True, then the evaluation options are printed
                 **estimator_params):
        #self.early_stopping_rounds = early_stopping_rounds
        self.eval_size = eval_size
        self.verbose = verbose
        if self.estimator is not None:
            self.set_params(eval_metric=eval_metric)
            self.set_params(**estimator_params)

    def set_params(self, **params):
        return self.estimator.set_params(**params)

    def get_params(self, **params):
        return self.estimator.get_params()

    def fit(self, X, y):
        x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=self.eval_size)
        self.estimator.fit(x_train,
                           y_train,
                           #early_stopping_rounds=self.early_stopping_rounds,
                           eval_set=[(x_val, y_val)],
                           verbose=self.verbose)

        return self

    def predict(self, X):
        return self.estimator.predict(X)


class XGBoostRegressorWithEarlyStop(XGBoostWithEarlyStop):
    def __init__(self, *args, **kwargs):
        self.estimator = XGBRegressor()
        super(XGBoostRegressorWithEarlyStop, self).__init__(*args, **kwargs)


# Get Data
data = dso.get_data_with_normal_deviation(de.read_data("House_Rent_Dataset.csv"), "Rent")
X, Y = de.split_features_and_target(data, "Rent")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=111)

# Configure Pre-Processors
numeric_transformer = SimpleImputer(strategy='mean')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessors = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, dso.only_numeric_column_names(X)),
        ("categorical", categorical_transformer, dso.only_non_numeric_column_names(X))
    ]
)

# Set up the pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessors),
    ('xgbr', XGBoostRegressorWithEarlyStop(#early_stopping_rounds=6,
                                           eval_size=0.2))
])

# Set parameters
param_grid = {
    'xgbr__n_estimators': [100, 500],
    'xgbr__learning_rate': [0.1, 0.05, 0.01],
    'xgbr__n_jobs': [4, 10],
    'xgbr__early_stopping_rounds': [5, 10]
}

# Fit grid, cv for Cross Validation
grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='neg_mean_absolute_error')
grid.fit(X, Y)

print("Mae on Test data: {}".format(mean_absolute_error(grid.predict(X_test), Y_test)))
print("Best score: {}".format(grid.best_score_))
print("Best params: {}".format(grid.best_params_))
print("Best estimator: {}".format(grid.best_estimator_))
