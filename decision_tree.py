from dataset_parser import parse_dataset
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report

def base_tree(data, output, test_data, test_output):
    tree = DecisionTreeClassifier(criterion="entropy").fit(data, output)
    prediction = tree.predict(test_data)
    #print(confusion_matrix(y_true=test_output, y_pred=prediction))
    print(classification_report(y_true=test_output, y_pred=prediction, zero_division=0))

# Param grid for best decision tree
param_grid = {
    'criterion' : ['gini'], # leaving it on only gini seems to increase the accuracy by 1-2%
    'max_depth' : [10, None], # Not much difference between these values
    "min_samples_split":[0.001], # After trying many numbers, this seems to be the sweet spot to not drop in performance
    "min_impurity_decrease":[0.0002], # After trying many numbers 0.0002 seems to be the sweet spot for an extra 1-2%
    'class_weight' : [None, 'balanced'] # Not much difference between these values
}

def best_tree(data, output, test_data, test_output):
    gridSearch = GridSearchCV(DecisionTreeClassifier(), param_grid)
    gridSearch.fit(data, output)
    prediction = gridSearch.predict(test_data)
    #print(confusion_matrix(y_true=test_output, y_pred=prediction))
    print(classification_report(y_true=test_output, y_pred=prediction, zero_division=0))


if __name__ == "__main__":
    X_train, y_train, X_validate, y_validate, X_test, y_test = parse_dataset(1)
    base_tree(X_train, y_train, X_test, y_test)
    best_tree(X_train, y_train, X_test, y_test)

    X_train, y_train, X_validate, y_validate, X_test, y_test = parse_dataset(2)
    base_tree(X_train, y_train, X_test, y_test)
    best_tree(X_train, y_train, X_test, y_test)

