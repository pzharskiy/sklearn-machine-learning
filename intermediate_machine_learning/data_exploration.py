import pandas as pd
from sklearn.model_selection import train_test_split

filepath_to_data_directory = "C:/Users/Pavel_Zharski/IdeaProjects/" \
                             "learning/python/sklearn-machine-learning/" \
                             "sklearn-machine-learning/data/house_renting/"

# Settings to show all columns, not limit them while using head()
pd.set_option('display.max_columns', None)


def read_data(filename="House_Rent_Dataset.csv",
              dropna=True,
              index_col=None,
              parse_dates=False,
              print_describe=False,
              print_columns=False):
    filepath = filepath_to_data_directory + filename
    data = pd.read_csv(filepath,
                   index_col=index_col,
                   parse_dates=parse_dates)
    if dropna:
        # {0 or ‘index’ for rows dropping, 1 or ‘columns’ for columns dropping}
        data = data.dropna(axis=0)
    if print_describe:
        print(data.describe())
    if print_columns:
        print(data.columns)
    return data

def split_features_and_target(data, target_column_name):
    Y = data[target_column_name]
    X = data.drop(target_column_name, axis=1)
    return X, Y


def prepare_data_sets(data,
                      features=None,
                      target_column_name="Price",
                      test_size=0.2):
    if features is None:
        features = ['Rooms', 'Bedroom2', 'Bathroom','Landsize', 'BuildingArea', 'Lattitude', 'Longtitude']

    y = data[target_column_name]
    X = data[features]

    return train_test_split(X, y, test_size=test_size, random_state=0)


def split_train_eval_test(X, Y, test_size=0.2, eval_size=0.1):
    X_train_eval, X_test, Y_train_eval, Y_test = train_test_split(X, Y, test_size=test_size, random_state=0)
    X_train, X_eval, Y_train, Y_eval = train_test_split(X_train_eval, Y_train_eval, test_size=eval_size, random_state=0)
    return X_train, X_eval, X_test, Y_train, Y_eval, Y_test


def print_data_head(melbourne_data):
    print(melbourne_data.head())