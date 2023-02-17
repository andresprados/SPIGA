import numpy as np

# Demo libs
from spiga.demo.analyze.features.basic import ObjectAnalyzed


class Face(ObjectAnalyzed):

    def __init__(self):
        super().__init__()
        self.bbox = np.zeros(5)
        self.key_landmarks = - np.ones((5, 2))
        self.landmarks = None
        self.face_id = -1
        self.past_states = []
        self.num_past_states = 5




