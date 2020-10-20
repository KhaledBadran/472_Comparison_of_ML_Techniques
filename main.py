from classifiers import Classifiers
import sys

if __name__ == "__main__":

    dataset_index = [1, 2]
    for dataset in dataset_index:
        classifiers = Classifiers(dataset)
        clfs = {
            'GNB': classifiers.GNB,
            'Base_DT': classifiers.Base_DT,
            'Best_DT': classifiers.Best_DT,
            'PER': classifiers.PER,
            'Base_MLP': classifiers.Base_MLP,
            'Best_MLP': classifiers.Best_MLP,
        }
        try:
            if len(sys.argv) == 1:  # No extra argument passed, Run all
                for key, value in clfs.items():
                    print(f"Running: {key} for dataset {dataset}")
                    clfs[key]()
            else:
                print(f"Running: {sys.argv[1]} for dataset {dataset}")
                clfs[sys.argv[1]]()
        except:
            print(f"Supported parameter list:\n{clfs.keys()}")
            print("Example: python main.py GNB")
            sys.exit()
