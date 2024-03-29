from sign_lang_model_dynamic import SignLangModelDynamic
from sign_lang_model import SignLangModel
import torch
import numpy as np
from util import process_features, normalize_landmark_history, landmark_history_preprocess

class RecognitionModel():
    def __init__(self, modelPath, num_hands, type):
        """
            Creates a RecognitionModel object.

            Parameters:
            modelPath (str): Path of the .pth trained model file.
            num_hands (int): Number of hands that this model will detect.
            type (str): RecognitionModel type (dynamic or static).
        """
        self.model = torch.load(modelPath)
        self.model.eval()
        self.type = type
        self.num_hands = num_hands

    def __evaluate_static__(self, landmark_data, num_hands, should_reflect):
        features = process_features(landmark_data, should_reflect)
        if len(features) < 42 and num_hands == 2:
            return

        tensor = torch.from_numpy(np.array(features))
        tensor = tensor.to(torch.float32)

        results = self.model(tensor[None, ...])
        result_arr = results.detach().numpy()
        result = np.argmax(result_arr)
        confidence = 2**results[0][result].item()

        return (result, confidence)

    def __evaluate_dynamic__(self, landmark_history, num_hands, should_reflect):
        landmark_history = normalize_landmark_history(landmark_history, should_reflect)
        compressed = landmark_history_preprocess(landmark_history, num_hands)

        print(compressed)

        tensor = torch.from_numpy(np.array(compressed))
        tensor = tensor.to(torch.float32)

        results = self.model(tensor[None, ...])

        result_arr = results.detach().numpy()

        print(results)
        result = np.argmax(result_arr)
        confidence = 2**results[0][result].item()

        return (result, confidence)
    
    def evaluate(self, landmark_data, should_reflect=False):
        """
            Evaluates the landmark data with the currently loaded model.

            Parameters:
            landmark_data (list): 
                For static recognition, this is an input of 2x21 (for one-hand recognition) or 2x42 (for two-hand recognition), consisting of
                landmark points.
                For dynamic recognition, this is an input of 30x2x21 (for one-hand recognition) or 30x2x42 (for two-hand recognition),
                consisting of landmark history over the course of 30 frames.
            should_reflect (boolean):
                This flag should be true if the hand being recognized is the user's left-hand (Mediapipe recognizes it as the right-hand, but
                if the user is signing directly towards the webcam, it is their left-hand).
            
            Returns:
            Tuple of (result, confidence)
        """
        if self.type == "static":
            return self.__evaluate_static__(landmark_data, self.num_hands, should_reflect)
        else:
            return self.__evaluate_dynamic__(landmark_data, self.num_hands, should_reflect)