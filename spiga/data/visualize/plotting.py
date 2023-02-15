import matplotlib.pyplot as plt
import numpy as np
import cv2

import spiga.data.loaders.augmentors.utils as dlu

BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
PURPLE = (128, 0, 128)


def draw_landmarks(image, landmarks, visible=None, mask=None, thick_scale=1, colors=(GREEN, RED)):
    # Fix variable
    thick = int(2 * thick_scale + 0.5)
    # Initialize variables if need it
    if visible is None:
        visible = np.ones(len(landmarks))
    if mask is None:
        mask = np.ones(len(landmarks))

    mask = np.array(mask, dtype=bool)
    visible = np.array(visible, dtype=bool)

    # Clean and split landmarks
    landmarks = landmarks[mask]
    visible = visible[mask]
    ldm_vis = landmarks[visible]
    not_visible = np.logical_not(visible)
    ldm_notvis = landmarks[not_visible]

    # Plot landmarks
    if image.shape[0] == 3:
        image = image.transpose(1, 2, 0)

    canvas = image.copy()
    canvas = _write_circles(canvas, ldm_vis, color=colors[0], thick=thick)
    canvas = _write_circles(canvas, ldm_notvis, color=colors[1], thick=thick)

    return canvas


def _write_circles(canvas, landmarks, color=RED, thick=2):
    for xy in landmarks:
        xy = np.array(xy+0.5, dtype=int)
        canvas = cv2.circle(canvas, (xy[0], xy[1]), thick, color, -1)
    return canvas


def plot_landmarks_pil(image, landmarks, visible=None, mask=None):

    # Initialize variables if need it
    if visible is None:
        visible = np.ones(len(landmarks))
    if mask is None:
        mask = np.ones(len(landmarks))

    mask = np.array(mask, dtype=bool)
    visible = np.array(visible, dtype=bool)
    not_visible = np.logical_not(visible)

    # Clean and split landmarks
    landmarks = landmarks[mask]
    ldm_vis = landmarks[visible]
    ldm_notvis = landmarks[not_visible]

    # Plot landmarks
    if image.shape[0] == 3:
        image = image.transpose(1, 2, 0)

    plt.imshow(image / 255)
    plt.scatter(ldm_vis[:, 0], ldm_vis[:, 1], s=10, marker='.', c='g')
    plt.scatter(ldm_notvis[:, 0], ldm_notvis[:, 1], s=10, marker='.', c='r')
    plt.show()


def draw_pose(img, rot, trl, K, euler=False, size=0.5, colors=(BLUE, GREEN, RED)):
    if euler:
        rot = dlu.euler_to_rotation_matrix(rot)

    canvas = img.copy()
    rotV, _ = cv2.Rodrigues(rot)
    points = np.float32([[size, 0, 0], [0, -size, 0], [0, 0, -size], [0, 0, 0]]).reshape(-1, 3)
    axisPoints, _ = cv2.projectPoints(points, rotV, trl, K, (0, 0, 0, 0))
    axisPoints = axisPoints.astype(int)
    canvas = cv2.line(canvas, tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()), colors[0], 3)
    canvas = cv2.line(canvas, tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()), colors[1], 3)
    canvas = cv2.line(canvas, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()), colors[2], 3)

    return canvas


def enhance_heatmap(heatmap):
    map_aux = heatmap - heatmap.min()
    map_aux = map_aux / map_aux.max()
    map_img = cv2.applyColorMap((map_aux * 255).astype(np.uint8), cv2.COLORMAP_BONE)
    return map_img
