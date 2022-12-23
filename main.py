import pandas

import data.dataset as dataset
from classifiers import random_forest

data: pandas.DataFrame = dataset.get_data()
results: list = random_forest.predict(data)

for result in results:
    print(result.get_accuracy())
    result.plot_confusion_matrix()
