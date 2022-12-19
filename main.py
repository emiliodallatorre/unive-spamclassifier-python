import pandas

import classifiers.svm_radial as svm_polynomial
import data.dataset as dataset

data: pandas.DataFrame = dataset.get_data()
results: list = svm_polynomial.predict(data)

for result in results:
    print(result.get_accuracy())
    result.plot_confusion_map()
