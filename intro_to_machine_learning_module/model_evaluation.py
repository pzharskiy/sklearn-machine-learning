from sklearn.metrics import mean_absolute_error


def evaluate(trained_model, X_eval, y_eval):
    predicted = trained_model.predict(X_eval)
    mae = mean_absolute_error(y_eval, predicted)
    print("Mean Absolute Error:  {:.2f} for model {}".format(mae, trained_model.__class__.__name__))
    return mae