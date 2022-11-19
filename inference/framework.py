import os
import copy
import torch
import numpy as np

from models.spiga import SPIGA

import inference.pretreatment as pretreat
from inference.config import ModelConfig

# Paths
root_path = os.path.realpath(__file__).split('/inference/framework.py')[0]
weights_path_dft = root_path + '/models/weights/'


class SPIGAFramework:

    def __init__(self, model_cfg: ModelConfig(), gpus=[0]):

        # Parameters
        self.model_cfg = model_cfg
        self.gpus = gpus

        # Pretreatment initialization
        self.transforms = pretreat.get_transformers(self.model_cfg)

        # SPIGA model
        self.model_inputs = ['image', "model3d", "cam_matrix"]
        self.model = SPIGA(num_landmarks=model_cfg.num_landmarks, num_edges=model_cfg.num_edges)

        # Load weights and set model
        weights_path = self.model_cfg.model_weights_path
        if weights_path is None:
            weights_path = weights_path_dft
        weights_file = os.path.join(weights_path, self.model_cfg.model_weights)
        self.model.load_state_dict(torch.load(weights_file))
        self.model = self.model.cuda(gpus[0])
        self.model.eval()
        print('SPIGA model loaded!')

    def inference(self, image, bboxes):
        """
        @param self:
        @param image: Raw image
        @param bboxes: List of bounding box founded on the image [[x,y,w,h],...]
        @return: features dict {'landmarks': list with shape (num_bbox, num_landmarks, 2) and x,y referred to image size
                                'headpose': list with shape (num_bbox, 6) euler->[:3], trl->[3:]
        """
        batch_crops, crop_bboxes = self.pretreat(image, bboxes)
        outputs = self.net_forward(batch_crops)
        features = self.postreatment(outputs, crop_bboxes, bboxes)
        return features

    def pretreat(self, image, bboxes):
        crop_bboxes = []
        batch_crops = []
        for bbox in bboxes:
            sample = {'image': copy.deepcopy(image),
                      'bbox': copy.deepcopy(bbox)}
            sample_out = self.transforms(sample)
            crop_bboxes.append(sample_out['bbox'])
            batch_crops.append(sample_out['image'])

        return torch.tensor(batch_crops, dtype=torch.float), crop_bboxes

    def net_forward(self, image):
        outputs = self.model(image)
        return outputs

    def postreatment(self, output, crop_bboxes, bboxes):
        features = {}
        crop_bboxes = np.array(crop_bboxes)
        bboxes = np.array(bboxes)

        if 'Landmarks' in output.keys():
            landmarks = output['Landmarks'][-1].cpu().detach().numpy()
            landmarks = landmarks.transpose((1, 0, 2))
            landmarks = landmarks*self.model_cfg.image_size
            landmarks_norm = (landmarks - crop_bboxes[:, 0:2]) / crop_bboxes[:, 2:4]
            landmarks_out = (landmarks_norm * bboxes[:, 2:4]) + bboxes[:, 0:2]
            landmarks_out = landmarks_out.transpose((1, 0, 2))
            features['landmarks'] = landmarks_out.tolist()

        # Pose output
        if 'Pose' in output.keys():
            pose = output['Pose'].cpu().detach().numpy()
            features['headpose'] = pose.tolist()

        return features

    def select_inputs(self, batch):
        inputs = []
        for ft_name in self.model_inputs:
            data = batch[ft_name]
            inputs.append(self._data2device(data.type(torch.float)))
        return inputs

    def _data2device(self, data):
        if isinstance(data, list):
            data_var = data
            for data_id, v_data in enumerate(data):
                data_var[data_id] = self._data2device(v_data)
        else:
            with torch.no_grad():
                data_var = data.cuda(device=self.gpus[0], non_blocking=True)
        return data_var