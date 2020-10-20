from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from dataset_parser import parse_dataset
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron


class Classifiers:

    def __init__(self, dataset: int):
        self.dataset = dataset
        self.X_train, self.y_train, self.X_validate, self.y_validate, self.X_test, self.y_test = parse_dataset(
            dataset)

    '''
    This method creates a Gaussian Naive Bayes Classifier with default parameter values. It also tests
    the model and stores the results
    '''

    def GNB(self):
        classifier = GaussianNB()

        # Train the model
        classifier.fit(self.X_train, self.y_train)

        # Test the model and save the results
        prediction = classifier.predict(self.X_test)
        self.save_results(prediction, model_name='GNB',
                          classifier=classifier, dataset=self.dataset)

        return

    '''
    This method creates a Decision Tree Classifier with default parameter values. It also tests
    the model and stores the results
    '''

    def Base_DT(self):
        # Train the model
        classifier = DecisionTreeClassifier(
            criterion="entropy").fit(self.X_train, self.y_train)

        # Test the model and save the results
        prediction = classifier.predict(self.X_test)
        self.save_results(prediction, model_name='Base-DT',
                          classifier=classifier, dataset=self.dataset)
        return

    '''
    Best_DT: This method creates a better performing Decision Tree found by performing grid search to find the best combination
    of hyper-parameters. It also tests the model and stores the results.
    '''

    def Best_DT(self):
        # Param grid for best decision tree
        param_grid = {
            # leaving it on only gini seems to increase the accuracy by 1-2%
            'criterion': ['gini'],
            # Not much difference between these values
            'max_depth': [10, None],
            # After trying many numbers, this seems to be the sweet spot to not drop in performance
            "min_samples_split": [0.001],
            # After trying many numbers 0.0002 seems to be the sweet spot for an extra 1-2%
            "min_impurity_decrease": [0.0002],
            # Not much difference between these values
            'class_weight': [None, 'balanced']
        }

        # Train the model
        classifier = GridSearchCV(DecisionTreeClassifier(), param_grid)
        classifier.fit(self.X_train, self.y_train)

        # Test the model and save the results
        prediction = classifier.predict(self.X_test)
        self.save_results(prediction, model_name='Best-DT',
                          classifier=classifier, dataset=self.dataset)

        return

    """
    PER: This method implements a Perceptron with default parameter values.It also tests the model and stores the results.
    """

    def PER(self):

        # Train the model
        classifier = Perceptron()
        classifier.fit(self.X_train, self.y_train)

        # Test the model and save the results
        prediction = classifier.predict(self.X_test)
        self.save_results(prediction, model_name='PER',
                          classifier=classifier, dataset=self.dataset)

        return

    """
    Base-MLP: A baseline Multi-Layered Perceptron with 1 hidden layer of 100 neurons,
    sigmoid/logisticas activation function, stochastic gradient descent,
    and default values for the rest of the parameters.
    """

    def Base_MLP(self):

        # Train the model
        classifier = MLPClassifier(
            hidden_layer_sizes=(100,),
            activation='logistic',
            solver='sgd',
        ).fit(self.X_train, self.y_train)

        # Test the model and save the results
        prediction = classifier.predict(self.X_test)
        self.save_results(prediction, model_name='Base-MLP',
                          classifier=classifier, dataset=self.dataset)

        return

    def Best_MLP(self):
        # I try different parameters inside mlp.py. These were found to be the most optimal.
        classifier = MLPClassifier(
            hidden_layer_sizes=(150,),
            activation='logistic',
            solver='adam',
            max_iter=1000,
        ).fit(self.X_train, self.y_train)
        prediction = classifier.predict(self.X_test)
        self.save_results(prediction, model_name='Best-MLP',
                          classifier=classifier, dataset=self.dataset)
        return

    def save_results(self, y_predicted, model_name, classifier, dataset):

        # Create and save model prediction
        prediction_df = pd.DataFrame(y_predicted)
        prediction_df.to_csv(
            f'./results/{model_name}-DS{dataset}.csv', header=False)

        # Append confusion matrix to file
        matrix = pd.DataFrame(confusion_matrix(
            y_true=self.y_test, y_pred=y_predicted))
        matrix.to_csv(f'./results/{model_name}-DS{dataset}.csv', mode='a')

        # Plot confusion matrix
        plot_confusion_matrix(classifier, self.X_test,
                              self.y_test, cmap=plt.cm.Blues, normalize='true')
        plt.title(f'Confusion matrix for {model_name} classifier')
        plt.savefig(f'./results/{model_name}-DS{dataset}_confusion_matrix')

        # Create and save classification report
        print(classification_report(y_true=self.y_test,
                                    y_pred=y_predicted, zero_division=0))
        report = classification_report(
            y_true=self.y_test, y_pred=y_predicted, zero_division=0, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(
            f'./results/{model_name}-DS{dataset}_classification_report.csv')
        report_df.to_csv(
            f'./results/{model_name}-DS{dataset}.csv', mode='a')  # appending results to desired file
