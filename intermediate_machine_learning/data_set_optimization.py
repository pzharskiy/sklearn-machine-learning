import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder


def only_numeric(data):
    return data.select_dtypes(exclude=['object'])


def only_non_numeric(data):
    return data.select_dtypes(include=['object'])


def only_numeric_column_names(data):
    return data.columns[data.dtypes != 'object']


def only_non_numeric_column_names(data):
    return data.columns[data.dtypes == 'object']


def investigating_cardinality(data, columns, printable=False):
    if printable:
        print("---------------------------------\n"
              "Investigating cardinality\n"
              "---------------------------------")

    # Columns that will be one-hot encoded
    low_cardinality_cols = [col for col in columns if data[col].nunique() < 10]

    # Columns that will be dropped from the dataset
    high_cardinality_cols = list(set(columns) - set(low_cardinality_cols))

    if printable:
        print('Categorical columns that will be one-hot encoded:', low_cardinality_cols)
        print('\nCategorical columns that will be dropped from the dataset:', high_cardinality_cols)

    return low_cardinality_cols,high_cardinality_cols


def get_number_of_unique_entries_in_each_column(data, columns):
    # Get number of unique entries in each column
    numbers_of_unique_entries = [data[col].nunique() for col in columns] # or implementation using map(): list(map(lambda col: X_train[col].nunique(), object_cols))
    dictionary = dict(zip(columns, numbers_of_unique_entries))
    # Sort dictionary by values
    return sorted(dictionary.items(), key=lambda item: item[1])


def drop_columns_with_categorical_data(data):
    return data.drop(columns=only_non_numeric_column_names(data))


def apply_ordinal_encoding(X_train, X_test, printable=False):
    if printable:
        print("---------------------------------\n"
              "Ordinal encoding\n"
              "---------------------------------")

    # Categorical columns in the training data
    object_cols = only_non_numeric_column_names(X_train)

    # Columns that can be safely ordinal encoded
    good_label_cols = [col for col in object_cols if
                       set(X_test[col]).issubset(set(X_train[col]))]

    # Problematic columns that will be dropped from the dataset
    bad_label_cols = list(set(object_cols)-set(good_label_cols))
    if printable:
        print('Categorical columns that will be ordinal encoded:', good_label_cols)
        print('\nCategorical columns that will be dropped from the dataset:', bad_label_cols)

    # Drop categorical columns that will not be encoded
    label_X_train = X_train.drop(bad_label_cols, axis=1)
    label_X_test = X_test.drop(bad_label_cols, axis=1)

    # Apply ordinal encoder
    ordinal_encoder = OrdinalEncoder()
    label_X_train[good_label_cols] = ordinal_encoder.fit_transform(X_train[good_label_cols])
    label_X_test[good_label_cols] = ordinal_encoder.transform(X_test[good_label_cols])
    return label_X_train, label_X_test


def apply_one_hot_encoding(X_train, X_test, printable=False):
    if printable:
        print("---------------------------------\n"
              "One-hot encoding\n"
              "---------------------------------")

    # Categorical columns in the training data
    categorical_columns = only_non_numeric_column_names(X_train)

    # Investigate which categorical columns are suitable for one-hot encoding
    low_cardinality_cols,high_cardinality_cols = investigating_cardinality(X_train, categorical_columns)

    # Apply one-hot encoder to each column with categorical data
    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[low_cardinality_cols]))
    OH_cols_test = pd.DataFrame(OH_encoder.transform(X_test[low_cardinality_cols]))

    # Produce Readable Feature names, instead of generated numbers.
    # If not apply, then columns will be 0,1,2,3,4,5, If apply, columns will be 'City_Bangalore', 'City_Chennai', 'City_Delhi', 'City_Hyderabad'
    OH_cols_train.columns = OH_encoder.get_feature_names_out(input_features=X_train[low_cardinality_cols].columns)
    OH_cols_test.columns = OH_encoder.get_feature_names_out(input_features=X_test[low_cardinality_cols].columns)

    # One-hot encoding removed index; put it back
    OH_cols_train.index = X_train.index
    OH_cols_test.index = X_test.index

    if printable:
        print("One-hot encoded columns: {}".format(OH_cols_train.columns))

    # Remove categorical columns (will replace with one-hot encoding)
    num_X_train = X_train.drop(categorical_columns, axis=1)
    num_X_test = X_test.drop(categorical_columns, axis=1)

    # Add one-hot encoded columns to numerical features
    OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
    OH_X_valid = pd.concat([num_X_test, OH_cols_test], axis=1)

    if printable:
        print("After one-hot encoding the data.columns will look like: {}".format(OH_X_train.columns))

    return OH_X_train, OH_X_valid


def get_data_with_normal_deviation(data, Y_column_name='Rent'):
    return data[(data[Y_column_name] < data[Y_column_name].mean()+3*data[Y_column_name].std()) &
         (data[Y_column_name] > data[Y_column_name].mean()-3*data[Y_column_name].std())]












