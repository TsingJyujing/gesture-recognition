from dataclasses import dataclass

import numpy
import tensorflow as tf

from gesture_recognizer.util import non_max_suppression_fast


@dataclass
class BlazePalmDetectResult:
    # Left
    x: int
    y: int
    width: int
    height: int
    # keypoints: 2*7 matrix
    keypoints: numpy.ndarray


class BlazePalm:
    def __init__(self, model_path: str, anchors: numpy.ndarray, batch_size: int = 1):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        input_details = self.interpreter.get_input_details()
        self._input_index = input_details[0]["index"]

        output_details = self.interpreter.get_output_details()
        self._regressors_index = output_details[0]["index"]
        self._classificators_index = output_details[1]["index"]
        if batch_size != 1:
            self.interpreter.resize_tensor_input(self._input_index, [batch_size, 256, 256, 3])
        self.interpreter.allocate_tensors()


        self.anchors = anchors

    def get_box(self, input_data, prob_threshold: float = 0.5):
        self._inference(input_data)
        return self._extract_boxes(prob_threshold=prob_threshold)

    def extract_tensor(self, index: int):
        return self.interpreter.get_tensor(index)

    def _inference(self, input_data):
        """
        Inference model
        :param input_data: numpy ndarray which size is nx256x256x3, dtype=float32, range from -1.0~1.0
        :return:
        """
        self.interpreter.set_tensor(self._input_index, input_data)
        self.interpreter.invoke()

    def _extract_boxes(self, prob_threshold: float):
        regressors = self.interpreter.get_tensor(self._regressors_index)
        classificators = self.interpreter.get_tensor(self._classificators_index)
        P = 1 / (1.0 + numpy.exp(-classificators[0, :, 0]))  # P = sigmoid(classificators)
        detection_mask = P > prob_threshold
        candidate_detect = regressors[0, detection_mask, :]
        candidate_anchors = self.anchors[detection_mask, :]
        P_keep = P[detection_mask]
        moved_candidate_detect = candidate_detect.copy()
        moved_candidate_detect[:, :2] = candidate_detect[:, :2] + (candidate_anchors[:, :2] * 256)
        box_ids = non_max_suppression_fast(moved_candidate_detect, P_keep)
        results = []
        for i, box_id in enumerate(box_ids):
            dx, dy, w, h = candidate_detect[box_id, :4]
            center_wo_offst = candidate_anchors[box_id, :2] * 256
            center_x, center_y = center_wo_offst
            x = dx + center_x - 0.5 * w
            y = dy + center_y - 0.5 * h
            keypoints = center_wo_offst + candidate_detect[box_id, 4:].reshape(-1, 2)
            results.append(BlazePalmDetectResult(
                x=x, y=y,
                width=w, height=h,
                keypoints=keypoints
            ))
        return results
