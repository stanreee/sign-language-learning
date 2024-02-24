import cv2
from classifier import Classifier
from util import extract_features, process_features
import os
import csv

class StaticClassifier(Classifier):

    def __init__(self, hands) -> None:
        super().__init__("static", hands)
        pass

    def save(self, data, id, num_hands):
        for i in range(len(data)):
            data[i] = [id] + data[i]
        cur_dir = os.curdir
        fileName = str(self.name)
        # if num_hands > 1: fileName += "_2"
        with open(cur_dir + "/datasets/" + fileName + ".csv", 'a', encoding="UTF8", newline='') as f:
            writer = csv.writer(f, delimiter=',')
            # data.sort(key=lambda x: x[0])
            for row in data:
                writer.writerow([i for i in row])
        print("Frames saved for id", id, ".")
        
    
        