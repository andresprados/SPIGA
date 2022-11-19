import numpy as np


class Heatmaps:

    def __init__(self, num_maps, map_size, sigma, stride=1, norm=True):
        self.num_maps = num_maps
        self.sigma = sigma
        self.double_sigma_pw2 = 2*sigma*sigma
        self.doublepi_sigma_pw2 = self.double_sigma_pw2 * np.pi
        self.stride = stride
        self.norm = norm

        if isinstance(map_size, (tuple, list)):
            self.width = map_size[0]
            self.height = map_size[1]
        else:
            self.width = map_size
            self.height = map_size

        grid_x = np.arange(self.width) * stride + stride / 2 - 0.5
        self.grid_x = np.repeat(grid_x.reshape(1, self.width), self.num_maps, axis=0)
        grid_y = np.arange(self.height) * stride + stride / 2 - 0.5
        self.grid_y = np.repeat(grid_y.reshape(1, self.height), self.num_maps, axis=0)

    def __call__(self, sample):
        landmarks = sample['landmarks']
        landmarks = landmarks[-self.num_maps:]

        # Heatmap generation
        exp_x = np.exp(-(self.grid_x - landmarks[:, 0].reshape(-1, 1)) ** 2 / self.double_sigma_pw2)
        exp_y = np.exp(-(self.grid_y - landmarks[:, 1].reshape(-1, 1)) ** 2 / self.double_sigma_pw2)
        heatmaps = np.matmul(exp_y.reshape(self.num_maps, self.height, 1), exp_x.reshape(self.num_maps, 1, self.width))

        if self.norm:
            heatmaps = heatmaps/self.doublepi_sigma_pw2

        sample['heatmap2D'] = heatmaps
        return sample
