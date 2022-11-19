import cv2
import numpy as np

from data.loaders.augmentors.modern_posit import PositPose
from data.loaders.augmentors.heatmaps import Heatmaps
from data.loaders.augmentors.boundary import AddBoundary
from data.loaders.augmentors.landmarks import HorizontalFlipAug, RSTAug, OcclusionAug, \
                                              LightingAug, BlurAug, TargetCropAug


def get_transformers(data_config):

    # Data augmentation
    aug_names = data_config.aug_names
    augmentors = []

    if 'flip' in aug_names:
        augmentors.append(HorizontalFlipAug(data_config.database.ldm_flip_order, data_config.hflip_prob))
    if 'rotate_scale' in aug_names:
        augmentors.append(RSTAug(data_config.angle_range, data_config.scale_min,
                                 data_config.scale_max, data_config.trl_ratio))
    if 'occlusion' in aug_names:
        augmentors.append(OcclusionAug(data_config.occluded_min_len,
                                       data_config.occluded_max_len,
                                       data_config.database.num_landmarks))
    if 'lighting' in aug_names:
        augmentors.append(LightingAug(data_config.hsv_range_min, data_config.hsv_range_max))
    if 'blur' in aug_names:
        augmentors.append(BlurAug(data_config.blur_prob, data_config.blur_kernel_range))

    # Crop mandatory
    augmentors.append(TargetCropAug(data_config.image_size, data_config.ftmap_size, data_config.target_dist))
    # Opencv style
    augmentors.append(ToOpencv())

    # Gaussian heatmaps
    if 'heatmaps2D' in aug_names:
        augmentors.append(Heatmaps(data_config.database.num_landmarks, data_config.ftmap_size,
                                   data_config.sigma2D, norm=data_config.heatmap2D_norm))

    if 'boundaries' in aug_names:
        augmentors.append(AddBoundary(num_landmarks=data_config.database.num_landmarks,
                                      map_size=data_config.ftmap_size,
                                      sigma=data_config.sigmaBD))
    # Pose generator
    if data_config.generate_pose:
        augmentors.append(PositPose(data_config.database.ldm_ids,
                                    focal_ratio=data_config.focal_ratio,
                                    selected_ids=data_config.posit_ids,
                                    max_iter=data_config.posit_max_iter))

    return augmentors


class ToOpencv:
    def __call__(self, sample):
        # Convert in a numpy array and change to GBR
        image = np.array(sample['image'])
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        sample['image'] = image
        return sample
