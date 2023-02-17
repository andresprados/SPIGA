# SPIGA library
import spiga.inference.config as model_cfg
from spiga.inference.framework import SPIGAFramework

# Demo modules
import spiga.demo.analyze.extract.processor as pr


class SPIGAProcessor(pr.Processor):

    def __init__(self,
                 dataset='wflw',
                 features=('lnd', 'pose'),
                 gpus=[0]):

        super().__init__()

        # Configure and load processor
        self.processor_cfg = model_cfg.ModelConfig(dataset)
        self.processor = SPIGAFramework(self.processor_cfg, gpus=gpus)

        # Define attributes
        if 'lnd' in features:
            self.attributes.append('landmarks')
            self.attributes.append('landmarks_ids')
        if 'pose' in features:
            self.attributes.append('headpose')

    def process_frame(self, frame, tracked_obj):
        bboxes = []
        for obj in tracked_obj:
            x1, y1, x2, y2 = obj.bbox[:4]
            bbox_wh = [x1, y1, x2-x1, y2-y1]
            bboxes.append(bbox_wh)
        features = self.processor.inference(frame, bboxes)

        for obj_idx in range(len(tracked_obj)):
            # Landmarks output
            if 'landmarks' in self.attributes:
                tracked_obj[obj_idx].landmarks = features['landmarks'][obj_idx]
                tracked_obj[obj_idx].landmarks_ids = self.processor_cfg.dataset.ldm_ids
            # Headpose output
            if 'headpose' in self.attributes:
                tracked_obj[obj_idx].headpose = features['headpose'][obj_idx]

        return tracked_obj

    def plot_features(self, image, features, plotter, show_attributes):

        if 'landmarks' in self.attributes and 'landmarks' in show_attributes:
            x1, y1, x2, y2 = features.bbox[:4]
            thick = int(plotter.landmarks.thickness['lnd'] * (x2-x1)/200 + 0.5)
            if thick == 0:
                thick = 1
            image = plotter.landmarks.draw_landmarks(image, features.landmarks, thick=thick)

        if 'headpose' in self.attributes and 'headpose' in show_attributes:
            image = plotter.hpose.draw_headpose(image, features.bbox[:5],
                                                features.headpose[:3], features.headpose[3:], euler=True)
        return image
