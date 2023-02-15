import cv2
import numpy as np

from spiga.data.visualize.inspect_dataset import DatasetInspector, inspect_parser


class HeatmapInspector(DatasetInspector):

    def __init__(self, database, anns_type, data_aug=True, image_shape=(256,256)):

        super().__init__(database, anns_type, data_aug=data_aug, pose=False, image_shape=image_shape)

        self.data_config.aug_names.append('heatmaps2D')
        self.data_config.heatmap2D_norm = False
        self.data_config.aug_names.append('boundaries')
        self.data_config.shuffle = False
        self.reload_dataset()

    def show_dataset(self, ids_list=None):

        if ids_list is None:
            ids = self.get_idx(shuffle=self.data_config.shuffle)
        else:
            ids = ids_list

        for img_id in ids:
            data_dict = self.dataset[img_id]

            crop_imgs, _ = self.plot_features(data_dict)

            # Plot landmark crop
            cv2.imshow('crop', crop_imgs['lnd'])

            # Plot landmarks 2D (group)
            crop_allheats = self._plot_heatmaps2D(data_dict)

            # Plot boundaries shape
            cv2.imshow('boundary', np.max(data_dict['boundary'], axis=0))

            for lnd_idx in range(self.data_config.database.num_landmarks):
                # Heatmaps 2D
                crop_heats = self._plot_heatmaps2D(data_dict, lnd_idx)
                maps = cv2.hconcat([crop_allheats['heatmaps2D'], crop_heats['heatmaps2D']])
                cv2.imshow('heatmaps', maps)

                key = cv2.waitKey()
                if key == ord('q'):
                    break
                if key == ord('n'):
                    break

            if key == ord('q'):
                break

    def _plot_heatmaps2D(self, data_dict, heatmap_id=None):

        # Variables
        heatmaps = {}
        image = data_dict['image']

        if heatmap_id is None:
            heatmaps2D = data_dict['heatmap2D']
            heatmaps2D = np.max(heatmaps2D, axis=0)
        else:
            heatmaps2D = data_dict['heatmap2D'][heatmap_id]

        # Plot maps
        heatmaps['heatmaps2D'] = self._merge_imgmap(image, heatmaps2D)
        return heatmaps

    def _merge_imgmap(self, image, maps):
        crop_maps = cv2.applyColorMap(np.uint8(255 * maps), cv2.COLORMAP_JET)
        return cv2.addWeighted(image, 0.7, crop_maps, 0.3, 0)


if __name__ == '__main__':

    args = inspect_parser()
    data_aug = True
    database = args.database
    anns_type = args.anns
    select_img = args.img
    if args.clean:
        data_aug = False

    if len(args.shape) != 2:
        raise ValueError('--shape requires two values: width and height. Ej: --shape 256 256')
    else:
        img_shape = tuple(args.shape)

    visualizer = HeatmapInspector(database, anns_type, data_aug, image_shape=img_shape)
    visualizer.show_dataset(ids_list=select_img)
