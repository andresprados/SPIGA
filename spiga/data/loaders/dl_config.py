import os
import json
import pkg_resources
from collections import OrderedDict

# Default data paths
db_img_path = pkg_resources.resource_filename('spiga', 'data/databases')
db_anns_path = pkg_resources.resource_filename('spiga', 'data/annotations') + "/{database}/{file_name}.json"

class AlignConfig:

    def __init__(self, database_name, mode='train'):
        # Dataset
        self.database_name = database_name
        self.working_mode = mode
        self.database = None    # Set at self._update_database()
        self.anns_file = None   # Set at self._update_database()
        self.image_dir = None   # Set at self._update_database()
        self._update_database()
        self.image_size = (256, 256)
        self.ftmap_size = (256, 256)

        # Dataloaders
        self.ids = None         # List of a subset if need it
        self.shuffle = True     # Shuffle samples
        self.num_workers = 4    # Threads

        # Posit
        self.generate_pose = True       # Generate pose parameters from landmarks
        self.focal_ratio = 1.5          # Camera matrix focal length ratio
        self.posit_max_iter = 100       # Refinement iterations

        # Subset of robust ids in the 3D model to use in posit.
        # 'None' to use all the available model landmarks.
        self.posit_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                          14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

        # Data augmentation
        # Control augmentations with the following list, crop to self.img_size is mandatory, check target_dist param.
        if mode == 'train':
            self.aug_names = ['flip', 'rotate_scale', 'occlusion', 'lighting', 'blur']
        else:
            self.aug_names = []
            self.shuffle = False

        # Flip
        self.hflip_prob = 0.5
        # Rotation
        self.angle_range = 45.
        # Scale
        self.scale_max = 0.15
        self.scale_min = -0.15
        # Translation
        self.trl_ratio = 0.05       # Translation augmentation
        # Crop target rescale
        self.target_dist = 1.6      # Target distance zoom in/out around face. Default: 1.
        # Occlusion
        self.occluded_max_len = 0.4
        self.occluded_min_len = 0.1
        self.occluded_covar_ratio = 2.25**0.5
        # Lighting
        self.hsv_range_min = [-0.5, -0.5, -0.5]
        self.hsv_range_max = [0.5, 0.5, 0.5]
        # Blur
        self.blur_prob = 0.5
        self.blur_kernel_range = [0, 2]

        # Heatmaps 2D
        self.sigma2D = 1.5
        self.heatmap2D_norm = False

        # Boundaries
        self.sigmaBD = 1

    def update(self, params_dict):
        state_dict = self.state_dict()
        for k, v in params_dict.items():
            if k in state_dict or hasattr(self, k):
                setattr(self, k, v)
            else:
                Warning('Unknown option: {}: {}'.format(k, v))
        self._update_database()

    def state_dict(self, tojson=False):
        state_dict = OrderedDict()
        for k in self.__dict__.keys():
            if not k.startswith('_'):
                if tojson and k in ['database']:
                    continue
                state_dict[k] = getattr(self, k)
        return state_dict

    def _update_database(self):
        self.database = DatabaseStruct(self.database_name)
        self.anns_file = db_anns_path.format(database=self.database_name, file_name=self.working_mode)
        self.image_dir = self._get_imgdb_path()

    def _get_imgdb_path(self):
        img_dir = None
        if self.database_name in ['300wpublic', '300wprivate']:
            img_dir = db_img_path + '/300w/'
        elif self.database_name in ['aflw19', 'merlrav']:
            img_dir = db_img_path + '/aflw/data/'
        elif self.database_name in ['cofw', 'cofw68']:
            img_dir = db_img_path + '/cofw/'
        elif self.database_name in ['wflw']:
            img_dir = db_img_path + '/wflw/'
        return img_dir

    def __str__(self):
        state_dict = self.state_dict()
        text = 'Dataloader {\n'
        for k, v in state_dict.items():
            if isinstance(v, DatabaseStruct):
                text += '\t{}: {}'.format(k, str(v).expandtabs(12))
            else:
                text += '\t{}: {}\n'.format(k, v)
        text += '\t}\n'
        return text


class DatabaseStruct:

    def __init__(self, database_name):

        self.name = database_name
        self.ldm_ids, self.ldm_flip_order, self.ldm_edges_matrix = self._get_database_specifics()
        self.num_landmarks = len(self.ldm_ids)
        self.num_edges = len(self.ldm_edges_matrix[0])-1
        self.fields = ['imgpath', 'bbox', 'headpose', 'ids', 'landmarks', 'visible']

    def _get_database_specifics(self):
        '''Returns specifics ids and horizontal flip reorder'''

        database_name = self.name
        db_info_file = db_anns_path.format(database=database_name, file_name='db_info')
        ldm_edges_matrix = None

        if os.path.exists(db_info_file):
            with open(db_info_file) as jsonfile:
                db_info = json.load(jsonfile)

            ldm_ids = db_info['ldm_ids']
            ldm_flip_order = db_info['ldm_flip_order']
            if 'ldm_edges_matrix' in db_info.keys():
                ldm_edges_matrix = db_info['ldm_edges_matrix']

        else:
            raise ValueError('Database ' + database_name + 'specifics not defined. Missing db_info.json')

        return ldm_ids, ldm_flip_order, ldm_edges_matrix

    def state_dict(self):
        state_dict = OrderedDict()
        for k in self.__dict__.keys():
            if not k.startswith('_'):
                state_dict[k] = getattr(self, k)

        return state_dict

    def __str__(self):
        state_dict = self.state_dict()
        text = 'Database {\n'
        for k, v in state_dict.items():
            text += '\t{}: {}\n'.format(k, v)
        text += '\t}\n'
        return text



