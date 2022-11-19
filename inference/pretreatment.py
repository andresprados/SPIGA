import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import cv2


def get_transformers(data_config):
    transformer_seq = [
        Opencv2Pil(),
        TargetCropAug(data_config.image_size, data_config.target_dist),
        Pil2Opencv(),
        NormalizeAndPermute()]
    return transforms.Compose(transformer_seq)


class NormalizeAndPermute:
    def __call__(self, sample):
        image = np.array(sample['image'], dtype=np.float)
        image = np.transpose(image, (2, 0, 1))
        sample['image'] = image / 255
        return sample


class Opencv2Pil:
    def __call__(self, sample):
        image = cv2.cvtColor(sample['image'], cv2.COLOR_BGR2RGB)
        sample['image'] = Image.fromarray(image)
        return sample


class Pil2Opencv:
    def __call__(self, sample):
        image = np.array(sample['image'])
        sample['image'] = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return sample


class TargetCropAug:
    def __init__(self, new_size=128, target_dist=1.3):

        if isinstance(new_size, tuple):
            self.new_size_x = new_size[0]
            self.new_size_y = new_size[1]
        else:
            self.new_size_x = new_size
            self.new_size_y = new_size

        self.target_dist = target_dist

    def __call__(self, sample):
        x, y, w, h = sample['bbox']

        side = max(w, h) * self.target_dist
        x -= (side - w) / 2
        y -= (side - h) / 2

        # center of the enlarged bounding box
        x0, y0 = x + side / 2, y + side / 2
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
                                  [0, mu_y, new_y0 - mu_y * y0]], dtype=float)

        # Transformation works with PIL
        sample = self.map_affine_transformation(sample, affine_transf, (new_w, new_h))
        return sample

    def map_affine_transformation(self, sample, affine_transf, new_size=None):

        sample['image'] = self._image_affine_trans(sample['image'], affine_transf, new_size)
        sample['bbox'] = self._bbox_affine_trans(sample['bbox'], affine_transf)
        return sample

    def _image_affine_trans(self, image, affine_transf, new_size=None):

        if not new_size:
            new_size = image.size

        inv_affine_transf = self.get_inverse_transf(affine_transf)
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

    @staticmethod
    def get_inverse_transf(affine_transf):
        A = affine_transf[0:2, 0:2]
        b = affine_transf[:, 2]

        inv_A = np.linalg.inv(A)  # we assume A invertible!

        inv_affine = np.zeros((2, 3))
        inv_affine[0:2, 0:2] = inv_A
        inv_affine[:, 2] = -inv_A.dot(b)

        return inv_affine