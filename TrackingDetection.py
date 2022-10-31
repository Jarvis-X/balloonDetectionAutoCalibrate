import numpy as np

class TrackingDetection:
    """
    The class to process the raw detection results from the camera and opencv
    """
    # TODO: populate this class to accommodate the raw balloon tracking results and provide proper filtering
    def __init__(self, input_data):
        self.avg = 0
        self.data = input_data

    def _filter(self):
        self.avg = np.mean(self.data)

    def update(self, new_detections):
        self.data.pop(0)
        self.data.append(new_detections)
        self._filter()

    def print(self):
        print(self.data)

    def get(self):
        return self.avg

