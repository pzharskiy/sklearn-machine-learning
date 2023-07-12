import pandas as pd
from sklearn.model_selection import train_test_split

filepath_to_data_directory = "C:/Users/Pavel_Zharski/IdeaProjects/" \
                             "learning/python/sklearn-machine-learning/" \
                             "sklearn-machine-learning/data/"
# Settings to show all columns, not limit them while using head()
pd.set_option('display.max_columns', None)


def read_data(filename="melb_data.csv",
              dropna=True,
              print_describe=False,
              print_columns=False):
    filepath = filepath_to_data_directory + filename
    melbourne_data = pd.read_csv(filepath)
    if dropna:
        # {0 or ‘index’ for rows dropping, 1 or ‘columns’ for columns dropping}
        melbourne_data = melbourne_data.dropna(axis=0)
    if print_describe:
        print(melbourne_data.describe())
    if print_columns:
        print(melbourne_data.columns)
    return melbourne_data


def prepare_data_sets(melbourne_data,
                      melbourne_features=None,
                      melbourne_target_column_name="Price",
                      test_size=0.2):
    if melbourne_features is None:
        melbourne_features = ['Rooms', 'Bedroom2', 'Bathroom','Landsize', 'BuildingArea', 'Lattitude', 'Longtitude']

    y = melbourne_data[melbourne_target_column_name]
    X = melbourne_data[melbourne_features]

    return train_test_split(X, y, test_size=test_size, random_state=0)


def print_data_head(melbourne_data):
    print(melbourne_data.head())
