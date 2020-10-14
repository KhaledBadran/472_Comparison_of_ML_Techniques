import pandas as pd
import os

'''
Parameters
----------
dataset_number : int
    select either dataset 1 or 2.
'''


def parse_dataset(dataset_number: int):
    data_folder = './dataset'

    training_data_file = f'train_{dataset_number}.csv'
    validation_data_file = f'val_{dataset_number}.csv'
    test_data_file = f'test_with_label_{dataset_number}.csv'

    # Create DataFrames for training, validation, and test sets
    training_df = pd.read_csv(os.path.join(data_folder, training_data_file))
    validation_df = pd.read_csv(os.path.join(data_folder, validation_data_file))
    test_df = pd.read_csv(os.path.join(data_folder, test_data_file))

    # Split the dataframes into X (list of bit values) and y (the class)
    X_train = training_df.iloc[:, :-1]
    y_train = training_df.iloc[:, -1:]

    X_validate = validation_df.iloc[:, :-1]
    y_validate = validation_df.iloc[:, -1:]

    X_test = test_df.iloc[:, :-1]
    y_test = test_df.iloc[:, -1:]

    return X_train, y_train.values.ravel(), X_validate, y_validate.values.ravel(), X_test, y_test.values.ravel()

