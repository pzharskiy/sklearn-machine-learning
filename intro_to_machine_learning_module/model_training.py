from sklearn.ensemble import RandomForestRegressor


def train_random_trees_model(X_train, y_train, random_state=1):
    # Define model. Specify a number for random_state to ensure same results each run
    melbourne_model = RandomForestRegressor(random_state=random_state)
    # Fit model
    melbourne_model.fit(X_train, y_train)
    return melbourne_model


def train(model_class, X_train, y_train):
    model = model_class.fit(X_train, y_train)
    return model