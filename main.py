import pandas

import classifiers.svm_linear as svm_linear
import data.dataset as dataset

data: pandas.DataFrame = dataset.get_data()
results: list = svm_linear.predict(data)

for result in results:
    print(result.get_accuracy())
    result.plot_confusion_map()
