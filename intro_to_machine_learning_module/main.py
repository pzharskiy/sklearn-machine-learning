import data_exploration
import model_training
import model_evaluation
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

models = [Lasso(),
          ElasticNet(),
          Ridge(),
          SVR(kernel="linear"),
          DecisionTreeRegressor(max_leaf_nodes=500, random_state=0),
          RandomForestRegressor(random_state=0),
          SVR(kernel="rbf")]

melbourne_data = data_exploration.read_data(print_describe=False)
X_train, X_eval, y_train, y_eval = data_exploration.prepare_data_sets(melbourne_data=melbourne_data)

for model in models:
    trained_model = model_training.train(model, X_train=X_train, y_train=y_train)
    model_evaluation.evaluate(trained_model, X_eval=X_eval, y_eval=y_eval)




