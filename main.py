import pandas as pd
import os
from typing import Tuple, List
from dataset_parser import parse_dataset
from sklearn.naive_bayes import GaussianNB

from classifiers import Classifiers


if __name__ == "__main__":
    X_train, y_train, X_validate, y_validate, X_test, y_test = parse_dataset(dataset_number=1)

    for dataset in [1, 2]:
        classifiers = Classifiers(dataset)

        # call the methods in the Classifiers instance
        classifiers.GNB()
        classifiers.Base_DT()
        classifiers.Best_DT()
        # classifiers.PER()
        # classifiers.Base_MLP()
        # classifiers.Best_MLP()
