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
        return confusion_matrix(self.y_test, self.y_predict)

    def plot_confusion_map(self):
        seaborn.heatmap(self.get_confusion_matrix(), annot=True, ax=plt.gca())
        plt.gca().set_title(f"Confusion matrix for {self.title}\naccuracy: {self.get_accuracy()}")
        plt.show()
