import pandas as pd
import os
from typing import Tuple, List
from dataset_parser import parse_dataset
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report

X_train, y_train, X_validate, y_validate, X_test, y_test = parse_dataset(dataset_number=1)

