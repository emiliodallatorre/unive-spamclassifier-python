import pandas

import data.dataset as dataset
from classifiers import k_nearest_neighbors

data: pandas.DataFrame = dataset.get_data()
results: list = k_nearest_neighbors.predict(data)

for result in results:
    print(result.get_accuracy())
    result.plot_confusion_matrix()
