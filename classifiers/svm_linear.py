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
    c: list = [1.0, 10.0, 100.0]

    results: list = []
    for i in range(len(c)):
        svcclassifier = SVC(kernel='linear', C=c[i])

        result, time = chrono_function(svcclassifier.fit, x_train, y_train)

        y_predict = svcclassifier.predict(x_test)
        score: np.ndarray = cross_val_score(svcclassifier, x, y, cv=5)

        results.append(ResultModel(y_test, y_predict, f"SVM Linear C={c[i]}", time, score))

    return results
