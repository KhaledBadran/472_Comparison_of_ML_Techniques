#
# MLP CLASSIFIER
# Docs: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
#

from dataset_parser import parse_dataset
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import preprocessing
import itertools


def base_mlp(data, output, test_data, test_output):
    """
    Base-MLP: A baseline Multi-Layered Perceptron with 1 hidden layer of 100 neurons,
    sigmoid/logisticas activation function, stochastic gradient descent,
    and default values for the rest of the parameters.
    """
    clf = MLPClassifier(
        hidden_layer_sizes=(100,),
        activation='logistic',
        solver='sgd',
    ).fit(X_train, y_train)
    prediction = clf.predict(test_data)
    print(f'Base MLP Prediction:\n{prediction}')
    print(
        f'Base MLP Classification report:\n{classification_report(y_true=y_test, y_pred=prediction, zero_division=0)}')
    print(
        f'Base MLP Confusion Matrix:\n{confusion_matrix(y_true=y_test, y_pred=prediction)}')


solvers = {
    "adam": "adam",
    "sgd": "sgd",
}

activations = {
    "identity": "identity",
    "sigmoid": "logistic",
    "tanh": "tanh",
    "relu": "relu",
}

networks = {
    "2_hidden_layers_low": (10, 10),
    "2_hidden_layers_mid": (30, 50),
    "3_hidden_layers_low": (10, 10, 10),
    "3_hidden_layers_mid": (30, 50, 80),
    "3_hidden_layers_high": (100, 100, 100),
}

mlp_param_grid = [
    {
        'activation': ['logistic', 'identity', 'tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'hidden_layer_sizes': [
            (100,), (100, 100,), (100, 100, 100,),
        ],
        'max_iter': [1000],
    }
]


def best_mlp(data, output, test_data, test_output):
    """
   Best-MLP:a better performing Multi-Layered Perceptron found by performing grid search to find the best combination of hyper-parameters.
   For this, you need to experiment with the following parameter values:
    • activation function:  sigmoid, tanh, relu and identity
    • 2 network architectures of your choice:  for eg 2 hidden layers with 30+50 nodes, 3 hidden layers with 10+10
    • solver:  Adam and stochastic gradient descent
    """
    clf = GridSearchCV(MLPClassifier(), mlp_param_grid).fit(X_train, y_train)

    print(f' Best parameters found: {clf.best_params_}')
    prediction = clf.predict(test_data)
    report = classification_report(
        y_true=y_test, y_pred=prediction, zero_division=0)
    print(f'Best MLP Prediction:\n{prediction}')
    print(
        f'Best MLP Classification report:\n{report}')
    print(
        f'Best MLP Confusion Matrix:\n{confusion_matrix(y_true=y_test, y_pred=prediction)}')


if __name__ == "__main__":
    X_train, y_train, X_validate, y_validate, X_test, y_test = parse_dataset(1)
    base_mlp(X_train, y_train, X_test, y_test)
    best_mlp(X_train, y_train, X_test, y_test)

    X_train, y_train, X_validate, y_validate, X_test, y_test = parse_dataset(2)
    base_mlp(X_train, y_train, X_test, y_test)
    best_mlp(X_train, y_train, X_test, y_test)
