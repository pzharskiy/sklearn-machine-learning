import pandas as pd

filepath_to_data_directory = "C:/Users/Pavel_Zharski/IdeaProjects/" \
                             "learning/python/" \
                             "sklearn-machine-learning/data/"


def read_data(filename, index_col=None, parse_dates=False, dropna=True):
    filepath = filepath_to_data_directory + filename
    data = pd.read_csv(filepath,
                       index_col=index_col,
                       parse_dates=parse_dates)
    data.dropna(axis=0)
    return data