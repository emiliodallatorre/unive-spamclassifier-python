import pandas

import data.dataset as dataset
from classifiers import naive_bayes_gaussian

data: pandas.DataFrame = dataset.get_data()
results: list = naive_bayes_gaussian.predict(data)

for result in results:
    print(result.get_accuracy())
    result.plot_confusion_matrix()
