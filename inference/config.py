from collections import OrderedDict


class ModelConfig(object):

    def __init__(self, dataset=None):
        # Model configuration
        self.model_weights = None
        self.model_weights_path = None
        # Pretreatment
        self.target_dist = 1.6
        self.image_size = (256, 256)
        # Outputs
        self.ftmap_size = (64, 64)
        # Dataset
        self.dataset = None
        self.num_landmarks = None
        self.num_edges = None

        if dataset is not None:
            self.update_with_dataset(dataset)

    def update_with_dataset(self, dataset):

        config_dict = {'dataset': dataset, 'model_weights': 'spiga_%s.pt'%dataset}
        if dataset in ['wflw']:
            config_dict['num_landmarks'] = 98
            config_dict['num_edges'] = 15
        elif dataset in ['300wpublic', '300wprivate', 'merlrav', 'cofw68']:
            config_dict['num_landmarks'] = 68
            config_dict['num_edges'] = 13
        else:
            raise NotImplementedError()

        if dataset == 'cofw68':     # Test only
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
