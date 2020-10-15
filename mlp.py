#
# MLP CLASSIFIER
# Docs: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
#

from dataset_parser import parse_dataset
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report

def base_mlp(data, output, test_data, test_output):
    """
    Base-MLP: A baseline Multi-Layered Perceptron with 1 hidden layer of 100 neurons,
    sigmoid/logisticas activation function, stochastic gradient descent, 
    and default values for the rest of the parameters.
    """
    clf = MLPClassifier(
        hidden_layer_sizes=100,
        activation='logistic',
        solver='sgd',
        ).fit(X_train, y_train)
    
    prediction = clf.predict(test_data)
    print(f'Prediction:\n{prediction}')
    print(f'Classification report:\n{classification_report(y_true=y_test, y_pred=prediction, zero_division=0)}')
    print(f'Confusion Matrix:\n{confusion_matrix(y_true=y_test, y_pred=prediction)}')
    

if __name__ == "__main__":
    X_train, y_train, X_validate, y_validate, X_test, y_test = parse_dataset(1)
    base_mlp(X_train, y_train, X_test, y_test)

    X_train, y_train, X_validate, y_validate, X_test, y_test = parse_dataset(2)
    base_mlp(X_train, y_train, X_test, y_test)

