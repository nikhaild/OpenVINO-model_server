from use_cases.use_case import UseCase
import numpy as np
import cv2

class ObjectDetection(UseCase):

    def supports_visualization() -> bool:
        return True


    def visualize(frame: np.ndarray, inference_result: np.ndarray) -> np.ndarray:
        CLASSES = ["Vehicle", "Person", "Bike"]
        COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        CONFIDENCE_THRESHOLD = 0.75

        # expecting inference_result in shape (1, 1, 200, 7)
        # ommiting batch dimension - frames are not analyzed in batches
        inference_result = inference_result[0][0]
        for single_prediction in inference_result:
            img_id = single_prediction[0]
            label = single_prediction[1]
            conf = single_prediction[2]
            x_min = single_prediction[3]
            y_min = single_prediction[4]
            x_max = single_prediction[5]
            y_max = single_prediction[6]

            if img_id != -1 and conf >= CONFIDENCE_THRESHOLD:
                height, width = frame.shape[0:2]
                cv2.rectangle(frame, (int(width * x_min), int(height * y_min)),
                    (int(width * x_max), int(height * y_max)), COLORS[int(label)], 1)
                cv2.putText(frame, str(CLASSES[int(label)]), (int(width * x_min)-10,
                    int(height * y_min)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    COLORS[int(label)], 1)
        return frame


    def preprocess(frame: np.ndarray) -> np.ndarray:
        return frame


    def postprocess(inference_result: np.ndarray):
        pass
