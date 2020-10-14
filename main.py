import pandas as pd
import os
from typing import Tuple, List
from dataset_parser import parse_dataset
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report

X_train, y_train, X_validate, y_validate, X_test, y_test = parse_dataset(dataset_number=1)

clf = GaussianNB()
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
print(prediction[:20])
print(classification_report(y_true=y_test, y_pred=prediction, zero_division=0))
print(confusion_matrix(y_true=y_test, y_pred=prediction))