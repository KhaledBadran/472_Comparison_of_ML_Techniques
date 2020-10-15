from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt
from pprint import pprint

class Classifiers():

    def __init__(self, X_train, y_train, X_validate, y_validate, X_test, y_test):

        self.X_train = X_train
        self.y_train = y_train
        self.X_validate = X_validate
        self.y_validate = y_validate
        self.X_test = X_test
        self.y_test = y_test


    '''
    This method creates a Gaussian Naive Bayes Classifier with default parameter values. It also tests
    the model and stores the results
    '''
    def GNB(self):

        classifier = GaussianNB()

        # Since no optimizations are needed here, we can add the validation set to the training set
        X_train_extended = self.X_train + self.X_validate
        y_train_extended = self.y_train + self.y_validate

        # Train the model
        classifier.fit(X_train_extended, y_train_extended)

        # Test the model and save the results
        prediction = classifier.predict(self.X_test)
        self.save_results(prediction, model_name='GNB', classifier=classifier)

        return

    def Base_DT(self):

        return

    def Best_DT(self):

        return

    def PER(self):

        return

    def Base_MLP(self):

        return

    def Best_MLP(self):

        return

    def save_results(self, y_predicted, model_name, classifier):

        # Create and save model prediction
        prediction_df = pd.DataFrame(y_predicted)
        prediction_df.to_csv(f'./results/{model_name}_prediction.csv')

        # Create and save classification report
        print(classification_report(y_true=self.y_test, y_pred=y_predicted, zero_division=0))
        report = classification_report(y_true=self.y_test, y_pred=y_predicted, zero_division=0, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(f'./results/{model_name}_classification_report.csv')

        # Plot confusion matrix
        plot_confusion_matrix(classifier, self.X_test, self.y_test, cmap=plt.cm.Blues, normalize='true')
        plt.title(f'Confusion matrix for {model_name} classifier')
        plt.show()

