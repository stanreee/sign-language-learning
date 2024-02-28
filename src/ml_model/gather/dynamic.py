from classifier import Classifier
import os
import csv
from sklearn.decomposition import PCA
import numpy as np
from gather_util import extract_features, process_features

class DynamicClassifier(Classifier):
    def __init__(self, hands) -> None:
        super().__init__("dynamic", hands)
        self.base_coords = None
        pass

    def save(self, data, id, num_hands):
        # reduce frame input dimensions from 30 to 18 (one hand) or 10 (two hands)
        dim = 18 if num_hands == 1 else 10
        pca = PCA(n_components=dim)
        pca.fit(data)
        dataToSave = [id]
        for i in range(pca.n_components_):
            dataToSave += np.ndarray.tolist(pca.components_[i])
        
        # save to file
        fileName = str(self.name)
        if num_hands > 1: fileName += "_2"
        cur_dir = os.curdir
        with open(cur_dir + "/datasets/" + fileName + ".csv", 'a', encoding="UTF8", newline='') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(dataToSave)
        print("Frames saved for id", id, ".")
    
    def capture(self, frame, frameNum, data):
        features, reflect = extract_features(frame, self.hands, self.num_hands)
        if len(features) >= 21 if self.num_hands == 1 else 42:
            # designate first landmark of the first frame as the base landmark
            # for each frame, all landmark coordinates will be relative to thsi base landmark
            if not self.base_coords:
                self.base_coords = features[0]
            features = process_features(features, reflect, self.base_coords)
            data.append(features)
        if frameNum >= self.FRAME_CAP:
            data, frameNum = self.endCapture(data, frameNum)
            self.base_coords = None
        return (data, frameNum)