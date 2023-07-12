from sklearn.datasets import load_diabetes
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
import pandas as pd


from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor, XGBClassifier


class XGBoostWithEarlyStop(BaseEstimator):

    def __init__(self,
                 early_stopping_rounds=5,
                 test_size=0.1,
                 eval_metric='mae',
                 **estimator_params):
        self.early_stopping_rounds = early_stopping_rounds
        self.test_size = test_size
        self.eval_metric=eval_metric='mae'
        if self.estimator is not None:
            self.set_params(**estimator_params)

    def set_params(self, **params):
        return self.estimator.set_params(**params)

    def get_params(self, **params):
        return self.estimator.get_params()

    def fit(self, X, y):
        x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=self.test_size)
        self.estimator.fit(x_train, y_train,
                           early_stopping_rounds=self.early_stopping_rounds,
                           eval_set=[(x_val, y_val)])
        return self

    def predict(self, X):
        return self.estimator.predict(X)

class XGBoostRegressorWithEarlyStop(XGBoostWithEarlyStop):
    def __init__(self, *args, **kwargs):
        self.estimator = XGBRegressor()
        super(XGBoostRegressorWithEarlyStop, self).__init__(*args, **kwargs)


class XGBoostClassifierWithEarlyStop(XGBoostWithEarlyStop):
    def __init__(self, *args, **kwargs):
        self.estimator = XGBClassifier()
        super(XGBoostClassifierWithEarlyStop, self).__init__(*args, **kwargs)


x, y = load_diabetes(return_X_y=True)
print(pd.DataFrame(x).head())

pipe = Pipeline([
    ('pca', PCA(5)),
    ('xgb', XGBoostRegressorWithEarlyStop())
])

param_grid = {
    'pca__n_components': [3, 5, 7],
    'xgb__n_estimators': [10, 20, 30, 50]
}

grid = GridSearchCV(pipe, param_grid, scoring='neg_mean_absolute_error')
grid.fit(x, y)
print(grid.best_params_)