from classifiers import Classifiers


if __name__ == "__main__":

    for dataset in [1, 2]:
        classifiers = Classifiers(dataset)

        # call the methods in the Classifiers instance
        # classifiers.GNB()
        # classifiers.Base_DT()
        # classifiers.Best_DT()
        # classifiers.PER()
        # classifiers.Base_MLP()
        classifiers.Best_MLP()
