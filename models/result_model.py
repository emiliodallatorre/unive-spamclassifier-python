import matplotlib.pyplot as plt
import numpy as np
import seaborn
from sklearn.metrics import confusion_matrix


class ResultModel:
    y_test: np.ndarray
    y_predict: np.ndarray
    title: str
    fitness_time: float
    score: np.ndarray

    def __init__(self, y_test: np.ndarray, y_predict: np.ndarray, title: str, fitness_time: float, score: np.ndarray):
        self.confusion_matrix = None
        self.y_test = y_test
        self.y_predict = y_predict
        self.title = title
        self.fitness_time = fitness_time
        self.score = score

    def get_misclassified(self) -> int:
        return (self.y_test != self.y_predict).sum()

    def get_accuracy(self) -> np.ndarray:
        return np.mean(self.score)

    def get_confusion_matrix(self) -> np.ndarray:
        if self.confusion_matrix is None:
            self.confusion_matrix = confusion_matrix(self.y_test, self.y_predict)

        return self.confusion_matrix

    def get_false_positives(self) -> int:
        return self.get_confusion_matrix()[0][1]

    def get_false_negatives(self) -> int:
        return self.get_confusion_matrix()[1][0]

    def plot_confusion_matrix(self):
        seaborn.heatmap(self.get_confusion_matrix(), annot=True, ax=plt.gca())
        plt.gca().set_title(f"Confusion matrix for {self.title}\nmean accuracy: {self.get_accuracy()}")
        plt.savefig(f"confusion_matrices/{self.title}.png")
        plt.show()
