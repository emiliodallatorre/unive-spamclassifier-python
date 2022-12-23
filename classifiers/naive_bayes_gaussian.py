import numpy as np
import pandas
from numpy import ndarray
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from models.result_model import ResultModel
from utils.benchmark import chrono_function


def bessel_corrected_variance(distribution: np.ndarray) -> ndarray:
    return np.var(distribution) * (len(distribution) / (len(distribution) - 1))


def gaussian_function(param, mean, sigma) -> float:
    if sigma == 0:
        return np.inf

    return (1 / np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(-0.5 * (((param - mean) / sigma) ** 2))


class NaiveBayesGaussian(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.classes = None
        self.freq = []
        self.var = []
        self.mean = []

    def fit(self, x, y):
        self.classes = np.unique(y)

        for c in self.classes:
            self.freq.append((y == c).sum() / y.shape[0])
            self.mean.append(x[y == c].mean(axis=0))
            self.var.append(bessel_corrected_variance(x[y == c]))

        # Get smallest variance greater than 0
        self.var = np.array(self.var)
        self.var[self.var == 0] = np.min(self.var[self.var > 0]) * 0.1

    def predict(self, x):
        size = x.shape[0]

        y = np.zeros(size, dtype=self.classes.dtype)

        for i in range(size):
            max_prob = 0
            max_c = 0
            for c in range(len(self.classes)):
                probs = float(self.norm(x.values[i], c) * self.freq[c])

                if probs > max_prob:
                    max_prob = probs
                    max_c = c
            y[i] = self.classes[max_c]
        return y

    def norm(self, document: list, target: int):
        document: np.array = np.array(document)

        probability: float = 1
        for i, field in enumerate(document):
            probability *= gaussian_function(field, self.mean[target][i], self.var[target][i])

        return probability


def predict(data: pandas.DataFrame) -> list:
    x = data.drop("spam", axis=1)
    y = data["spam"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

    results: list = []

    classifier = NaiveBayesGaussian()
    result, time = chrono_function(classifier.fit, x_train, y_train)

    y_predict = classifier.predict(x_test)
    score: np.ndarray = cross_val_score(classifier, x, y, cv=5)

    results.append(ResultModel(y_test, y_predict, f"Naive Bayes Gaussian", time, score))

    return results
