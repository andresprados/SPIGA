import random
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

# My libs
import spiga.data.loaders.augmentors.utils as dlu


class HorizontalFlipAug:
    def __init__(self, ldm_flip_order, prob=0.5):
        self.prob = prob
        self.ldm_flip_order = ldm_flip_order

    def __call__(self, sample):
        img = sample['image']
        landmarks = sample['landmarks']
        mask = sample['mask_ldm']
        vis = sample['visible']
        bbox = sample['bbox']

        if random.random() < self.prob:
            new_img = transforms.functional.hflip(img)

            lm_new_order = self.ldm_flip_order
            new_landmarks = landmarks[lm_new_order]
            new_landmarks = (new_landmarks - (img.size[0], 0)) * (-1, 1)
            new_mask = mask[lm_new_order]
            new_vis = vis[lm_new_order]

            x, y, w, h = bbox
            new_x = img.size[0] - x - w
            new_bbox = np.array((new_x, y, w, h))

            sample['image'] = new_img
            sample['landmarks'] = new_landmarks
            sample['mask_ldm'] = new_mask
            sample['visible'] = new_vis
            sample['bbox'] = new_bbox

        return sample


class GeometryBaseAug:

    def __call__(self, sample):
        raise NotImplementedError('Inheritance __call__ not defined')

    def map_affine_transformation(self, sample, affine_transf, new_size=None):
        sample['image'] = self._image_affine_trans(sample['image'], affine_transf, new_size)
        sample['bbox'] = self._bbox_affine_trans(sample['bbox'], affine_transf)
        if 'landmarks' in sample.keys():
            sample['landmarks'] = self._landmarks_affine_trans(sample['landmarks'], affine_transf)
        return sample

    def clean_outbbox_landmarks(self, shape, landmarks, mask):
        filter_x1 = landmarks[:, 0] >= shape[0]
        filter_x2 = landmarks[:, 0] < (shape[0] + shape[2])
        filter_x = np.logical_and(filter_x1,filter_x2)

        filter_y1 = landmarks[:, 1] >= shape[1]
        filter_y2 = landmarks[:, 1] < (shape[1] + shape[3])
        filter_y = np.logical_and(filter_y1, filter_y2)

        filter_bbox = np.logical_and(filter_x, filter_y)
        new_mask = mask*filter_bbox
        new_landmarks = (landmarks.T * new_mask).T
        new_landmarks = new_landmarks.astype(int).astype(float)
        return new_mask, new_landmarks

    def _image_affine_trans(self, image, affine_transf, new_size=None):

        if not new_size:
            new_size = image.size

        inv_affine_transf = dlu.get_inverse_transf(affine_transf)
        new_image = image.transform(new_size, Image.AFFINE, inv_affine_transf.flatten())
        return new_image

    def _bbox_affine_trans(self, bbox, affine_transf):

        x, y, w, h = bbox
        images_bb = []
        for point in ([x, y, 1], [x + w, y, 1],
                      [x, y + h, 1], [x + w, y + h, 1]):
            images_bb.append(affine_transf.dot(point))
        images_bb = np.array(images_bb)

        new_corner0 = np.min(images_bb, axis=0)
        new_corner1 = np.max(images_bb, axis=0)

        new_x, new_y = new_corner0
        new_w, new_h = new_corner1 - new_corner0
        new_bbox = np.array((new_x, new_y, new_w, new_h))
        return new_bbox

    def _landmarks_affine_trans(self, landmarks, affine_transf):

        homog_landmarks = dlu.affine2homogeneous(landmarks)
        new_landmarks = affine_transf.dot(homog_landmarks.T).T
        return new_landmarks


class RSTAug(GeometryBaseAug):

    def __init__(self, angle_range=45., scale_min=-0.15, scale_max=0.15, trl_ratio=0.05):
        self.scale_max = scale_max
        self.scale_min = scale_min
        self.angle_range = angle_range
        self.trl_ratio = trl_ratio

    def __call__(self, sample):
        x, y, w, h = sample['bbox']

        x0, y0 = x + w/2, y + h/2  # center of the face, which will be the center of the rotation

        # Bbox translation
        rnd_Tx = np.random.uniform(-self.trl_ratio, self.trl_ratio) * w
        rnd_Ty = np.random.uniform(-self.trl_ratio, self.trl_ratio) * h
        sample['bbox'][0] += rnd_Tx
        sample['bbox'][1] += rnd_Ty

        scale = 1 + np.random.uniform(self.scale_min, self.scale_max)
        angle = np.random.uniform(-self.angle_range, self.angle_range)

        similarity = dlu.get_similarity_matrix(angle, scale, center=(x0, y0))
        new_sample = self.map_affine_transformation(sample, similarity)
        return new_sample


class TargetCropAug(GeometryBaseAug):
    def __init__(self, img_new_size=128, map_new_size=128, target_dist=1.3):

        self.target_dist = target_dist
        self.new_size_x, self.new_size_y = self._convert_shapes(img_new_size)
        self.map_size_x, self.map_size_y = self._convert_shapes(map_new_size)
        self.img2map_scale = False

        # Mismatch between img shape and featuremap shape
        if self.map_size_x != self.new_size_x or self.map_size_y != self.new_size_y:
            self.img2map_scale = True
            self.map_scale_x = self.map_size_x / self.new_size_x
            self.map_scale_y = self.map_size_y / self.new_size_y
            self.map_scale_xx = self.map_scale_x * self.map_scale_x
            self.map_scale_xy = self.map_scale_x * self.map_scale_y
            self.map_scale_yy = self.map_scale_y * self.map_scale_y

    def _convert_shapes(self, new_size):
        if isinstance(new_size, (tuple, list)):
            new_size_x = new_size[0]
            new_size_y = new_size[1]
        else:
            new_size_x = new_size
            new_size_y = new_size
        return new_size_x, new_size_y

    def __call__(self, sample):
        x, y, w, h = sample['bbox']
        # we enlarge the area taken around the bounding box
        # it is neccesary to change the botton left point of the bounding box
        # according to the previous enlargement. Note this will NOT be the new
        # bounding box!
        # We return square images, which is neccesary since
        # all the images must have the same size in order to form batches
        side = max(w, h) * self.target_dist
        x -= (side - w) / 2
        y -= (side - h) / 2

        # center of the enlarged bounding box
        x0, y0 = x + side/2, y + side/2
        # homothety factor, chosen so the new horizontal dimension will
        # coincide with new_size
        mu_x = self.new_size_x / side
        mu_y = self.new_size_y / side

        # new_w, new_h = new_size, int(h * mu)
        new_w = self.new_size_x
        new_h = self.new_size_y
        new_x0, new_y0 = new_w / 2, new_h / 2

        # dilatation + translation
        affine_transf = np.array([[mu_x, 0, new_x0 - mu_x * x0],
                                  [0, mu_y, new_y0 - mu_y * y0]])

        sample = self.map_affine_transformation(sample, affine_transf,(new_w, new_h))
        if 'landmarks' in sample.keys():
            img_shape = np.array([0, 0, self.new_size_x, self.new_size_y])
            sample['landmarks_float'] = sample['landmarks']
            sample['mask_ldm_float'] = sample['mask_ldm']
            sample['landmarks'] = np.round(sample['landmarks'])
            sample['mask_ldm'], sample['landmarks'] = self.clean_outbbox_landmarks(img_shape, sample['landmarks'],
                                                                                   sample['mask_ldm'])

            if self.img2map_scale:
                sample = self._rescale_map(sample)
        return sample

    def _rescale_map(self, sample):

        # Rescale
        lnd_float = sample['landmarks_float']
        lnd_float[:, 0] = self.map_scale_x * lnd_float[:, 0]
        lnd_float[:, 1] = self.map_scale_y * lnd_float[:, 1]

        # Filter landmarks
        lnd = np.round(lnd_float)
        filter_x = lnd[:, 0] >= self.map_size_x
        filter_y = lnd[:, 1] >= self.map_size_y
        lnd[filter_x] = self.map_size_x - 1
        lnd[filter_y] = self.map_size_y - 1
        new_lnd = (lnd.T * sample['mask_ldm']).T
        new_lnd = new_lnd.astype(int).astype(float)

        sample['landmarks_float'] = lnd_float
        sample['landmarks'] = new_lnd
        sample['img2map_scale'] = [self.map_scale_x, self.map_scale_y]
        return sample



class OcclusionAug:
    def __init__(self, min_length=0.1, max_length=0.4, num_maps=1):
        self.min_length = min_length
        self.max_length = max_length
        self.num_maps = num_maps

    def __call__(self, sample):
        x, y, w, h = sample['bbox']
        image = sample['image']
        landmarks = sample['landmarks']
        vis = sample['visible']

        min_ratio = self.min_length
        max_ratio = self.max_length
        rnd_width = np.random.randint(int(w * min_ratio), int(w * max_ratio))
        rnd_height = np.random.randint(int(h * min_ratio), int(h * max_ratio))

        # (xi, yi) and (xf, yf) are, respectively, the lower left points of the
        # occlusion rectangle and the upper right point.
        xi = int(x + np.random.randint(0, w - rnd_width))
        xf = int(xi + rnd_width)
        yi = int(y + np.random.randint(0, h - rnd_height))
        yf = int(yi + rnd_height)

        pixels = np.array(image)
        pixels[yi:yf, xi:xf, :] = np.random.uniform(0, 255, size=3)
        image = Image.fromarray(pixels)
        sample['image'] = image

        # Update visibilities
        filter_x1 = landmarks[:, 0] >= xi
        filter_x2 = landmarks[:, 0] < xf
        filter_x = np.logical_and(filter_x1, filter_x2)

        filter_y1 = landmarks[:, 1] >= yi
        filter_y2 = landmarks[:, 1] < yf
        filter_y = np.logical_and(filter_y1, filter_y2)

        filter_novis = np.logical_and(filter_x, filter_y)
        filter_vis = np.logical_not(filter_novis)
        sample['visible'] = vis * filter_vis
        return sample


class LightingAug:
    def __init__(self, hsv_range_min=(-0.5, -0.5, -0.5), hsv_range_max=(0.5, 0.5, 0.5)):
        self.hsv_range_min = hsv_range_min
        self.hsv_range_max = hsv_range_max

    def __call__(self, sample):
        # Convert to HSV colorspace from RGB colorspace
        image = np.array(sample['image'])
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Generate new random values
        H = 1 + np.random.uniform(self.hsv_range_min[0], self.hsv_range_max[0])
        S = 1 + np.random.uniform(self.hsv_range_min[1], self.hsv_range_max[1])
        V = 1 + np.random.uniform(self.hsv_range_min[2], self.hsv_range_max[2])
        hsv[:, :, 0] = np.clip(H*hsv[:, :, 0], 0, 179)
        hsv[:, :, 1] = np.clip(S*hsv[:, :, 1], 0, 255)
        hsv[:, :, 2] = np.clip(V*hsv[:, :, 2], 0, 255)
        # Convert back to BGR colorspace
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        sample['image'] = Image.fromarray(image)

        return sample


class BlurAug:
    def __init__(self, blur_prob=0.5, blur_kernel_range=(0, 2)):
        self.blur_prob = blur_prob
        self.kernel_range = blur_kernel_range

    def __call__(self, sample):
        # Smooth image
        image = np.array(sample['image'])
        if np.random.uniform(0.0, 1.0) < self.blur_prob:
            kernel = np.random.random_integers(self.kernel_range[0], self.kernel_range[1]) * 2 + 1
            image = cv2.GaussianBlur(image, (kernel, kernel), 0, 0)
        sample['image'] = Image.fromarray(image)

        return sample




