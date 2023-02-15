from collections import OrderedDict

from spiga.data.loaders.dl_config import DatabaseStruct


class ModelConfig(object):

    def __init__(self, dataset_name=None):
        # Model configuration
        self.model_weights = None
        self.model_weights_path = None
        # Pretreatment
        self.focal_ratio = 1.5          # Camera matrix focal length ratio.
        self.target_dist = 1.6          # Target distance zoom in/out around face.
        self.image_size = (256, 256)
        # Outputs
        self.ftmap_size = (64, 64)
        # Dataset
        self.dataset = None

        if dataset_name is not None:
            self.update_with_dataset(dataset_name)

    def update_with_dataset(self, dataset_name):

        config_dict = {'dataset': DatabaseStruct(dataset_name),
                       'model_weights': 'spiga_%s.pt' % dataset_name}

        if dataset_name == 'cofw68':     # Test only
            config_dict['model_weights'] = 'spiga_300wprivate.pt'

        self.update(config_dict)

    def update(self, params_dict):
        state_dict = self.state_dict()
        for k, v in params_dict.items():
            if k in state_dict or hasattr(self, k):
                setattr(self, k, v)
            else:
                raise Warning('Unknown option: {}: {}'.format(k, v))

    def state_dict(self):
        state_dict = OrderedDict()
        for k in self.__dict__.keys():
            if not k.startswith('_'):
                state_dict[k] = getattr(self, k)
        return state_dict
