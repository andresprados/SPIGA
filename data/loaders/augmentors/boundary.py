import numpy as np
from scipy import interpolate
import cv2


class AddBoundary(object):
    def __init__(self, num_landmarks=68, map_size=64, sigma=1, min_dpi=64):
        self.num_landmarks = num_landmarks
        self.sigma = sigma

        if isinstance(map_size, (tuple, list)):
            self.width = map_size[0]
            self.height = map_size[1]
        else:
            self.width = map_size
            self.height = map_size

        if max(map_size) > min_dpi:
            self.dpi = max(map_size)
        else:
            self.dpi = min_dpi

        self.fig_size =[self.height/self.dpi, self.width/self.dpi]

    def __call__(self, sample):
        landmarks = sample['landmarks_float']
        mask_lnd = sample['mask_ldm_float']
        boundaries = self.get_dataset_boundaries(landmarks, mask_lnd)
        functions = {}

        for key, points in boundaries.items():
            if len(points) != 0:
                temp = points[0]
                new_points = points[0:1, :]
                for point in points[1:]:
                    if point[0] == temp[0] and point[1] == temp[1]:
                        continue
                    else:
                        new_points = np.concatenate((new_points, np.expand_dims(point, 0)), axis=0)
                        temp = point

                points = new_points
                if points.shape[0] == 1:
                    points = np.concatenate((points, points+0.001), axis=0)
                k = min(4, points.shape[0])
                functions[key] = interpolate.splprep([points[:, 0], points[:, 1]], k=k-1,s=0)

        boundary_maps = np.zeros((len(boundaries), self.height, self.width))
        for i_map, key in enumerate(functions.keys()):
            boundary_map = np.zeros((self.height, self.width))
            xnew = np.arange(0, 1, 1/self.dpi)
            out = interpolate.splev(xnew, functions[key][0], der=0)

            out = np.round(out).astype(int).transpose()
            out = out[out[:, 0] < self.height]
            out = out[out[:, 1] < self.width]
            boundary_map[out[:,1], out[:,0]]= 255

            # Smooth
            sigma = self.sigma
            temp = 255 - boundary_map.astype(np.uint8)
            temp = cv2.distanceTransform(temp, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            temp = temp.astype(np.float32)
            temp = np.where(temp < 3*sigma, np.exp(-(temp*temp)/(2*sigma*sigma)), 0 )
            boundary_maps[i_map] = temp

        sample['boundary'] = boundary_maps
        return sample

    def get_dataset_boundaries(self, landmarks, mask_lnd):
        boundaries = {}
        if self.num_landmarks == 68:
            cheek = landmarks[0:17]
            boundaries['cheek'] = cheek[mask_lnd[0:17] > 0]
            left_eyebrow = landmarks[17:22]
            boundaries['left_eyebrow'] = left_eyebrow[mask_lnd[17:22] > 0]
            right_eyebrow = landmarks[22:27]
            boundaries['right_eyebrow'] = right_eyebrow[mask_lnd[22:27] > 0]
            nose = landmarks[27:31]
            boundaries['nose'] = nose[mask_lnd[27:31] > 0]
            nose_bot = landmarks[31:36]
            boundaries['nose_bot'] = nose_bot[mask_lnd[31:36] > 0]
            uper_left_eyelid = landmarks[36:40]
            boundaries['upper_left_eyelid'] = uper_left_eyelid[mask_lnd[36:40] > 0]
            lower_left_eyelid = np.array([landmarks[i] for i in [36, 41, 40, 39]])
            lower_left_eyelid_mask = np.array([mask_lnd[i] for i in [36, 41, 40, 39]])
            boundaries['lower_left_eyelid'] = lower_left_eyelid[lower_left_eyelid_mask > 0]
            upper_right_eyelid = landmarks[42:46]
            boundaries['upper_right_eyelid'] = upper_right_eyelid[mask_lnd[42:46] > 0]
            lower_right_eyelid = np.array([landmarks[i] for i in [42, 47, 46, 45]])
            lower_right_eyelid_mask = np.array([mask_lnd[i] for i in [42, 47, 46, 45]])
            boundaries['lower_right_eyelid'] = lower_right_eyelid[lower_right_eyelid_mask > 0]
            upper_outer_lip = landmarks[48:55]
            boundaries['upper_outer_lip'] = upper_outer_lip[mask_lnd[48:55] > 0]
            lower_outer_lip = np.array([landmarks[i] for i in [48, 59, 58, 57, 56, 55, 54]])
            lower_outer_lip_mask = np.array([mask_lnd[i] for i in [48, 59, 58, 57, 56, 55, 54]])
            boundaries['lower_outer_lip'] = lower_outer_lip[lower_outer_lip_mask > 0]
            upper_inner_lip = np.array([landmarks[i] for i in [60, 61, 62, 63, 64]])
            upper_inner_lip_mask = np.array([mask_lnd[i] for i in [60, 61, 62, 63, 64]])
            boundaries['upper_inner_lip'] = upper_inner_lip[upper_inner_lip_mask > 0]
            lower_inner_lip = np.array([landmarks[i] for i in [60, 67, 66, 65, 64]])
            lower_inner_lip_mask = np.array([mask_lnd[i] for i in [60, 67, 66, 65, 64]])
            boundaries['lower_inner_lip'] = lower_inner_lip[lower_inner_lip_mask > 0]
            
        elif self.num_landmarks == 98:
            boundaries['cheek'] = landmarks[0:33]
            boundaries['upper_left_eyebrow'] = landmarks[33:38]
            boundaries['lower_left_eyebrow'] = np.array([landmarks[i] for i in [33, 41, 40, 39, 38]])
            boundaries['upper_right_eyebrow'] = landmarks[42:47]
            boundaries['lower_right_eyebrow'] = landmarks[46:51]
            boundaries['nose'] = landmarks[51:55]
            boundaries['nose_bot'] = landmarks[55:60]
            boundaries['upper_left_eyelid'] = landmarks[60:65]
            boundaries['lower_left_eyelid'] = np.array([landmarks[i] for i in [60, 67, 66, 65, 64]])
            boundaries['upper_right_eyelid'] = landmarks[68:73]
            boundaries['lower_right_eyelid'] = np.array([landmarks[i] for i in [68, 75, 74, 73, 72]])
            boundaries['upper_outer_lip'] = landmarks[76:83]
            boundaries['lower_outer_lip'] = np.array([landmarks[i] for i in [76, 87, 86, 85, 84, 83, 82]])
            boundaries['upper_inner_lip'] = np.array([landmarks[i] for i in [88, 89, 90, 91, 92]])
            boundaries['lower_inner_lip'] = np.array([landmarks[i] for i in [88, 95, 94, 93, 92]])
            
        return boundaries
