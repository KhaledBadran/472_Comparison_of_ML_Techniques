from dataset_parser import parse_dataset
from collections import defaultdict, OrderedDict
from matplotlib import pyplot as plt


def plot_classes(dataset: int):
    X_train, y_train, X_validate, y_validate, X_test, y_test = parse_dataset(dataset)
    freq = defaultdict(int)
    for value in y_train:
        freq[value] += 1

    # order the classes in ascending order
    ordered = OrderedDict(sorted(freq.items()))

    fig = plt.figure()
    ax = fig.add_axes([0.08, 0.05, 0.90, 0.94])
    ax.bar(ordered.keys(), ordered.values())
    fig.show()


if __name__ == "__main__":
    plot_classes(2)

