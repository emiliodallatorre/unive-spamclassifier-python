import numpy as np
from pandas import DataFrame
from tqdm import tqdm

import data.dataset as dataset
from classifiers import naive_bayes_gaussian, k_nearest_neighbors, random_forest, svm_radial, svm_polynomial, \
    svm_linear, svm_angular
from models.result_model import ResultModel

data: DataFrame = dataset.get_data()
classifiers: list = [
    naive_bayes_gaussian,
    k_nearest_neighbors,
    random_forest,
    svm_radial,
    svm_polynomial,
    svm_linear,
    svm_angular
]

raw_results: list[ResultModel] = []
for classifier in tqdm(classifiers):
    raw_results.extend(classifier.predict(data))

results: DataFrame = DataFrame(np.empty((0, 7)))
results.columns = ["Title", "Misclassified", "Accuracy", "False positives", "False negatives", "Time", "Mean score"]
for result in raw_results:
    result.plot_confusion_matrix()

    results = results.append({
        "Title": result.title,
        "Misclassified": result.get_misclassified(),
        "Accuracy": result.get_accuracy(),
        "False positives": result.get_false_positives(),
        "False negatives": result.get_false_negatives(),
        "Time": result.fitness_time,
        "Mean score": np.mean(result.score)
    }, ignore_index=True)

results.to_csv("results.csv")
