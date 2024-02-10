from classifier import Classifier
import os
from util import extract_features, process_features
import csv
from sklearn.decomposition import PCA
import numpy as np

class DynamicClassifier(Classifier):
    def __init__(self, hands) -> None:
        super().__init__("dynamic", hands)
        pass

    def save(self, data, id, num_hands):
        pca = PCA(n_components=18)
        pca.fit(data)
        dataToSave = [id]
        # print(pca.components_)
        # for i in range(pca.n_components_):
        #     print(type(np.ndarray.tolist(pca.components_[i])), type([id]))
        #     dataToSave.append([id] + np.ndarray.tolist(pca.components_[i]))
        fileName = str(self.name)
        if num_hands > 1: fileName += "_2"
        for i in range(pca.n_components_):
            dataToSave += np.ndarray.tolist(pca.components_[i])
        dataToSave = [dataToSave]
        cur_dir = os.curdir
        with open(cur_dir + "/datasets/" + fileName + ".csv", 'a', encoding="UTF8", newline='') as f:
            writer = csv.writer(f, delimiter=',')
            # data.sort(key=lambda x: x[0])
            for row in dataToSave:
                writer.writerow([i for i in row])
        print("Frames saved for id", id, ".")