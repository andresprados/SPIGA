import numpy as np
import cv2


class BasicLayout:

    # Variables
    colors = {'green': (0, 255, 0),
              'red': (0, 0, 255),
              'blue': (255, 0, 0),
              'purple': (128, 0, 128),
              'white': (255, 255, 255),
              'black': (0, 0, 0)}

    thickness_dft = {'circle': 2}

    def __init__(self):
        self.thickness = self.thickness_dft

    def draw_circles(self, canvas, coord_list, color=colors['red'], thick=None):
        if thick is None:
            thick = self.thickness['circle']

        for xy in coord_list:
            xy = np.array(xy + 0.5, dtype=int)
            canvas = cv2.circle(canvas, (xy[0], xy[1]), thick, color, -1)
        return canvas

    def update_thickness(self, thick_dict):
        for k, v in thick_dict.items():
            self.thickness[k] = v

    def reset_thickness(self):
        self.thickness = self.thickness_dft

    def update_thick_byratio(self, ratio_dict):
        for key, ratio in ratio_dict.items():
            self.thickness[key] = int(self.thickness_dft[key] * ratio + 0.5)


