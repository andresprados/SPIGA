import os
import pkg_resources
import numpy as np
import cv2

# My libs
from spiga.data.loaders.augmentors.utils import rotation_matrix_to_euler

# Model file nomenclature
model_file_dft = pkg_resources.resource_filename('spiga', 'data/models3D') + '/mean_face_3D_{num_ldm}.txt'


class PositPose:

    def __init__(self, ldm_ids, focal_ratio=1, selected_ids=None, max_iter=100,
                  fix_bbox=True, model_file=model_file_dft):

        # Load 3D face model
        model3d_world, model3d_ids = self._load_world_shape(ldm_ids, model_file)

        # Generate id mask to pick only the robust landmarks for posit
        if selected_ids is None:
            model3d_mask = np.ones(len(ldm_ids))
        else:
            model3d_mask = np.zeros(len(ldm_ids))
            for index, posit_id in enumerate(model3d_ids):
                if posit_id in selected_ids:
                    model3d_mask[index] = 1

        self.ldm_ids = ldm_ids                  # Ids from the database
        self.model3d_world = model3d_world      # Model data
        self.model3d_ids = model3d_ids          # Model ids
        self.model3d_mask = model3d_mask        # Model mask ids
        self.max_iter = max_iter                # Refinement iterations
        self.focal_ratio = focal_ratio          # Camera matrix focal length ratio
        self.fix_bbox = fix_bbox                # Camera matrix centered on image (False to centered on bbox)

    def __call__(self, sample):

        landmarks = sample['landmarks']
        mask = sample['mask_ldm']

        # Camera matrix
        img_shape = np.array(sample['image'].shape)[0:2]
        if 'img2map_scale' in sample.keys():
            img_shape = img_shape * sample['img2map_scale']

        if self.fix_bbox:
            img_bbox = [0, 0, img_shape[1], img_shape[0]]   # Shapes given are inverted (y,x)
            cam_matrix = self._camera_matrix(img_bbox)
        else:
            bbox = sample['bbox']   # Scale error when ftshape and img_shape mismatch
            cam_matrix = self._camera_matrix(bbox)

        # Save intrinsic matrix and 3D model landmarks
        sample['cam_matrix'] = cam_matrix
        sample['model3d'] = self.model3d_world

        world_pts, image_pts = self._set_correspondences(landmarks, mask)

        if image_pts.shape[0] < 4:
            print('POSIT does not work without landmarks')
            rot_matrix, trl_matrix = np.eye(3, dtype=float), np.array([0, 0, 0])
        else:
            rot_matrix, trl_matrix = self._modern_posit(world_pts, image_pts, cam_matrix)

        euler = rotation_matrix_to_euler(rot_matrix)
        sample['pose'] = np.array([euler[0], euler[1], euler[2], trl_matrix[0], trl_matrix[1], trl_matrix[2]])
        sample['model3d_proj'] = self._project_points(rot_matrix, trl_matrix, cam_matrix, norm=img_shape)
        return sample

    def _load_world_shape(self, ldm_ids, model_file):
        return load_world_shape(ldm_ids, model_file=model_file)

    def _camera_matrix(self, bbox):
        focal_length_x = bbox[2] * self.focal_ratio
        focal_length_y = bbox[3] * self.focal_ratio
        face_center = (bbox[0] + (bbox[2] * 0.5)), (bbox[1] + (bbox[3] * 0.5))

        cam_matrix = np.array([[focal_length_x, 0, face_center[0]],
                               [0, focal_length_y, face_center[1]],
                               [0, 0, 1]])
        return cam_matrix

    def _set_correspondences(self, landmarks, mask):
        # Correspondences using labelled and robust landmarks
        img_mask = np.logical_and(mask, self.model3d_mask)
        img_mask = img_mask.astype(bool)

        image_pts = landmarks[img_mask]
        world_pts = self.model3d_world[img_mask]
        return world_pts, image_pts

    def _modern_posit(self, world_pts, image_pts, cam_matrix):
        return modern_posit(world_pts, image_pts, cam_matrix, self.max_iter)

    def _project_points(self, rot, trl, cam_matrix, norm=None):
        # Perspective projection model
        trl = np.expand_dims(trl, 1)
        extrinsics = np.concatenate((rot, trl), 1)
        proj_matrix = np.matmul(cam_matrix, extrinsics)

        # Homogeneous landmarks
        pts = self.model3d_world
        ones = np.ones(pts.shape[0])
        ones = np.expand_dims(ones, 1)
        pts_hom = np.concatenate((pts, ones), 1)

        # Project landmarks
        pts_proj = np.matmul(proj_matrix, pts_hom.T).T
        pts_proj = pts_proj / np.expand_dims(pts_proj[:, 2], 1) # Lambda = 1

        if norm is not None:
            pts_proj[:, 0] /= norm[0]
            pts_proj[:, 1] /= norm[1]
        return pts_proj[:, :-1]


def load_world_shape(db_landmarks, model_file=model_file_dft):

    # Load 3D mean face coordinates
    num_ldm = len(db_landmarks)
    filename = model_file.format(num_ldm=num_ldm)
    if not os.path.exists(filename):
        raise ValueError('No 3D model find for %i landmarks' % num_ldm)

    posit_landmarks = np.genfromtxt(filename, delimiter='|', dtype=int, usecols=0).tolist()
    mean_face_3D = np.genfromtxt(filename, delimiter='|', dtype=(float, float, float), usecols=(1, 2, 3)).tolist()
    world_all = len(mean_face_3D)*[None]
    index_all = len(mean_face_3D)*[None]

    for cont, elem in enumerate(mean_face_3D):
        pt3d = [elem[2], -elem[0], -elem[1]]
        lnd_idx = db_landmarks.index(posit_landmarks[cont])
        world_all[lnd_idx] = pt3d
        index_all[lnd_idx] = posit_landmarks[cont]

    return np.array(world_all), np.array(index_all)


def modern_posit(world_pts, image_pts, cam_matrix, max_iters):
    # Homogeneous world points
    num_landmarks = image_pts.shape[0]
    one = np.ones((num_landmarks, 1))
    A = np.concatenate((world_pts, one), axis=1)
    B = np.linalg.pinv(A)

    # Normalize image points
    focal_length = cam_matrix[0,0]
    img_center = (cam_matrix[0,2], cam_matrix[1,2])
    centered_pts = np.zeros((num_landmarks,2))
    centered_pts[:,0] = (image_pts[:,0]-img_center[0])/focal_length
    centered_pts[:,1] = (image_pts[:,1]-img_center[1])/focal_length
    Ui = centered_pts[:,0]
    Vi = centered_pts[:,1]

    # POSIT loop
    Tx, Ty, Tz = 0.0, 0.0, 0.0
    r1, r2, r3 = [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]
    for iter in range(0, max_iters):
        I = np.dot(B,Ui)
        J = np.dot(B,Vi)

        # Estimate translation vector and rotation matrix
        normI = 1.0 / np.sqrt(I[0] * I[0] + I[1] * I[1] + I[2] * I[2])
        normJ = 1.0 / np.sqrt(J[0] * J[0] + J[1] * J[1] + J[2] * J[2])
        Tz = np.sqrt(normI * normJ)  # geometric average instead of arithmetic average of classicPosit
        r1N = I*Tz
        r2N = J*Tz
        r1 = r1N[0:3]
        r2 = r2N[0:3]
        r1 = np.clip(r1, -1, 1)
        r2 = np.clip(r2, -1, 1)
        r3 = np.cross(r1,r2)
        r3T = np.concatenate((r3, [Tz]), axis=0)
        Tx = r1N[3]
        Ty = r2N[3]

        # Compute epsilon, update Ui and Vi and check convergence
        eps = np.dot(A, r3T)/Tz
        oldUi = Ui
        oldVi = Vi
        Ui = np.multiply(eps, centered_pts[:,0])
        Vi = np.multiply(eps, centered_pts[:,1])
        deltaUi = Ui - oldUi
        deltaVi = Vi - oldVi
        delta = focal_length * focal_length * (np.dot(np.transpose(deltaUi), deltaUi) + np.dot(np.transpose(deltaVi), deltaVi))
        if iter > 0 and delta < 0.01:  # converged
            break

    rot_matrix = np.array([np.transpose(r1), np.transpose(r2), np.transpose(r3)])
    trl_matrix = np.array([Tx, Ty, Tz])
    # Convert to the nearest orthogonal rotation matrix
    w, u, vt = cv2.SVDecomp(rot_matrix)  # R = U*D*Vt
    rot_matrix = np.matmul(np.matmul(u, np.eye(3, dtype=float)), vt)
    return rot_matrix, trl_matrix

