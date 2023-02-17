import numpy as np

# Demo libs
from spiga.demo.visualize.layouts.plot_basics import BasicLayout


class LandmarkLayout(BasicLayout):

    BasicLayout.thickness_dft['lnd'] = 3

    def __init__(self):
        super().__init__()

    def draw_landmarks(self, image, landmarks, visible=None, mask=None,
                       thick=None, colors=(BasicLayout.colors['green'], BasicLayout.colors['red'])):

        # Initialize variables if need it
        if visible is None:
            visible = np.ones(len(landmarks))
        if mask is None:
            mask = np.ones(len(landmarks))
        if thick is None:
            thick = self.thickness['lnd']

        if isinstance(landmarks, (list, tuple)):
            landmarks = np.array(landmarks)
        if isinstance(visible, (list, tuple)):
            visible = np.array(visible)
        if isinstance(mask, (list, tuple)):
            mask = np.array(mask)

        # Clean and split landmarks
        ldm_vis, ldm_notvis = self._split_lnd_by_vis(landmarks, visible, mask)

        # PIL images to OpenCV
        if image.shape[0] == 3:
            image = image.transpose(1, 2, 0)

        # Plot landmarks
        canvas = self.draw_circles(image, ldm_vis, color=colors[0], thick=thick)
        canvas = self.draw_circles(canvas, ldm_notvis, color=colors[1], thick=thick)
        return canvas

    @ staticmethod
    def _split_lnd_by_vis(landmarks, visible, mask):
        mask = np.array(mask, dtype=bool)
        visible = np.array(visible, dtype=bool)
        landmarks = landmarks[mask]
        visible = visible[mask]
        ldm_vis = landmarks[visible]
        not_visible = np.logical_not(visible)
        ldm_notvis = landmarks[not_visible]
        return ldm_vis, ldm_notvis
