import os
import json
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from spiga.data.loaders.transforms import get_transformers


class AlignmentsDataset(Dataset):
    '''Loads datasets of images with landmarks and bounding boxes.
    '''

    def __init__(self,
                 database,
                 json_file,
                 images_dir,
                 image_size=(128, 128),
                 transform=None,
                 indices=None,
                 debug=False):
        """

        :param database: class DatabaseStruct containing all the specifics of the database

        :param json_file: path to the json file which contains the names of the images, landmarks, bounding boxes, etc

        :param images_dir: path of the directory containing the images.

        :param image_size: tuple like e.g. (128, 128)

        :param transform: composition of transformations that will be applied to the samples.

        :param debug_mode: bool if True, loads a very reduced_version of the dataset for debugging purposes.

        :param indices: If it is a list of indices, allows to work with the subset of
                        items specified by the list. If it is None, the whole set is used.
        """

        self.database = database
        self.images_dir = images_dir
        self.transform = transform
        self.image_size = image_size
        self.indices = indices
        self._imgs_dict = None
        self.debug = debug

        with open(json_file) as jsonfile:
            self.data = json.load(jsonfile)

    def __len__(self):
        '''Returns the length of the dataset
        '''
        if self.indices is None:
            return len(self.data)
        else:
            return len(self.indices)

    def __getitem__(self, sample_idx):
        '''Returns sample of the dataset of index idx'''

        # To allow work with a subset
        if self.indices is not None:
            sample_idx = self.indices[sample_idx]

        # Load sample image
        img_name = os.path.join(self.images_dir, self.data[sample_idx]['imgpath'])
        if not self._imgs_dict:
            image_cv = cv2.imread(img_name)
        else:
            image_cv = self._imgs_dict[sample_idx]

        # Some images are B&W. We make sure that any image has three channels.
        if len(image_cv.shape) == 2:
            image_cv = np.repeat(image_cv[:, :, np.newaxis], 3, axis=-1)

        # Some images have alpha channel
        image_cv = image_cv[:, :, :3]

        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image_cv)

        # Load sample anns
        ids = np.array(self.data[sample_idx]['ids'])
        landmarks = np.array(self.data[sample_idx]['landmarks'])
        bbox = np.array(self.data[sample_idx]['bbox'])
        vis = np.array(self.data[sample_idx]['visible'])
        headpose = self.data[sample_idx]['headpose']

        # Generate bbox if need it
        if bbox is None:
            # Compute bbox using landmarks
            aux = landmarks[vis == 1.0]
            bbox = np.zeros(4)
            bbox[0] = min(aux[:, 0])
            bbox[1] = min(aux[:, 1])
            bbox[2] = max(aux[:, 0]) - bbox[0]
            bbox[3] = max(aux[:, 1]) - bbox[1]

        # Clean and mask landmarks
        mask_ldm = np.ones(self.database.num_landmarks)
        if not self.database.ldm_ids == ids.tolist():
            new_ldm = np.zeros((self.database.num_landmarks, 2))
            new_vis = np.zeros(self.database.num_landmarks)
            xyv = np.hstack((landmarks, vis[np.newaxis,:].T))
            ids_dict = dict(zip(ids.astype(int).astype(str), xyv))

            for pos, identifier in enumerate(self.database.ldm_ids):
                if str(identifier) in ids_dict:
                    x, y, v = ids_dict[str(identifier)]
                    new_ldm[pos] = [x,y]
                    new_vis[pos] = v
                else:
                    mask_ldm[pos] = 0
            landmarks = new_ldm
            vis = new_vis

        sample = {'image': image,
                  'sample_idx': sample_idx,
                  'imgpath': img_name,
                  'ids_ldm': np.array(self.database.ldm_ids),
                  'bbox': bbox,
                  'bbox_raw': bbox,
                  'landmarks': landmarks,
                  'visible': vis.astype(np.float64),
                  'mask_ldm': mask_ldm,
                  'imgpath_local': self.data[sample_idx]['imgpath'],
                  }

        if self.debug:
            sample['landmarks_ori'] = landmarks
            sample['visible_ori'] = vis.astype(np.float64)
            sample['mask_ldm_ori'] = mask_ldm
            if headpose is not None:
                sample['headpose_ori'] = np.array(headpose)

        if self.transform:
            sample = self.transform(sample)

        return sample


def get_dataset(data_config, pretreat=None, debug=False):

    augmentors = get_transformers(data_config)
    if pretreat is not None:
        augmentors.append(pretreat)

    dataset = AlignmentsDataset(data_config.database,
                                data_config.anns_file,
                                data_config.image_dir,
                                image_size=data_config.image_size,
                                transform=transforms.Compose(augmentors),
                                indices=data_config.ids,
                                debug=debug)
    return dataset
