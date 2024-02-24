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
        dim = 18 if num_hands == 1 else 10
        pca = PCA(n_components=dim)
        pca.fit(data)
        # print(data, len(data))
        # print(pca.components_, pca.n_components, len(pca.components_))
        dataToSave = [id]
        # print(pca.components_)
        for i in range(pca.n_components_):
            # print(type(np.ndarray.tolist(pca.components_[i])), type([id]))
            # dataToSave.append([id] + np.ndarray.tolist(pca.components_[i]))
            dataToSave += np.ndarray.tolist(pca.components_[i])
        fileName = str(self.name)
        if num_hands > 1: fileName += "_2"
        # dataToSave = [dataToSave]
        cur_dir = os.curdir
        with open(cur_dir + "/datasets/" + fileName + ".csv", 'a', encoding="UTF8", newline='') as f:
            writer = csv.writer(f, delimiter=',')
            # data.sort(key=lambda x: x[0])
            # for row in dataToSave:
            #     writer.writerow([i for i in row])
            writer.writerow(dataToSave)
        print("Frames saved for id", id, ".")