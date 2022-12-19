import numpy as np
import pandas
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from models.result_model import ResultModel
from utils.benchmark import chrono_function


def predict(data: pandas.DataFrame) -> list:
    x = data.drop("spam", axis=1)
    y = data["spam"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
    regularization: list = [1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0, 10000000.0]

    results: list = []
    for c in regularization:
        classifier = SVC(kernel='rbf', C=c, gamma='auto')

        result, time = chrono_function(classifier.fit, x_train, y_train)

        y_predict = classifier.predict(x_test)
        score: np.ndarray = cross_val_score(classifier, x, y, cv=5)

        results.append(ResultModel(y_test, y_predict, f"SVM Linear C={c}", time, score))

    return results