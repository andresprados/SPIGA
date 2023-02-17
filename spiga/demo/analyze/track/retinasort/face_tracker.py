import numpy as np

# Third party algorithms. Implementation maintained by SPIGA authors.
import sort_tracker
import retinaface

# My libs
import spiga.demo.analyze.track.retinasort.config as cfg
import spiga.demo.analyze.track.tracker as tracker
import spiga.demo.analyze.features.face as ft_face


class RetinaSortTracker(tracker.Tracker):

    def __init__(self, config=cfg.cfg_retinasort):
        super().__init__()

        self.detector = retinaface.RetinaFaceDetector(model=config['retina']['model_name'],
                                                      extra_features=config['retina']['extra_features'],
                                                      cfg_postreat=config['retina']['postreat'])

        self.associator = sort_tracker.Sort(max_age=config['sort']['max_age'],
                                            min_hits=config['sort']['min_hits'],
                                            iou_threshold=config['sort']['iou_threshold'])
        self.obj_type = ft_face.Face
        self.attributes += ['bbox', 'face_id', 'key_landmarks']

    def process_frame(self, image, tracked_obj):
        # tracked_obj = []
        features = self.detector.inference(image)
        bboxes = features['bbox']
        bboxes = self._code_bbox_idx(bboxes)
        bboxes_id = self.associator.update(bboxes)
        bboxes_id, bbox_idx = self._decode_bbox_idx(bboxes_id)
        final_tracked_obj = []
        for idx, bbox in enumerate(bboxes_id):
            founded_flag = False
            for past_obj in tracked_obj:
                if past_obj.face_id == bbox[-1]:
                    past_obj.bbox = bbox[:5]
                    past_obj = self._update_extra_features(past_obj, features, bbox_idx[idx])
                    final_tracked_obj.append(past_obj)
                    tracked_obj.remove(past_obj)
                    founded_flag = True
                    break

            if not founded_flag:
                new_obj = self.obj_type()
                new_obj.bbox = bbox[:5]
                new_obj.face_id = bbox[5]
                new_obj = self._update_extra_features(new_obj, features, bbox_idx[idx])
                final_tracked_obj.append(new_obj)

        return final_tracked_obj

    def plot_features(self, image, features, plotter, show_attributes):
        if 'bbox' in show_attributes:
            image = plotter.bbox.draw_bbox(image, features.bbox)
        if 'face_id' in show_attributes:
            text_id = 'Face Id: %i' % features.face_id
            image = plotter.bbox.draw_bbox_text(image, features.bbox, text_id, offset=(0, -10), color=plotter.basic.colors['blue'])
            image = plotter.bbox.draw_bbox_line(image, features.bbox)
        return image

    def _code_bbox_idx(self, bboxes):
        bboxes = np.array(bboxes)
        bboxes[:, 4] += (np.arange(len(bboxes)) - 0.001)
        return bboxes

    def _decode_bbox_idx(self, bboxes):
        bboxes = np.array(bboxes)
        idx = bboxes[:, 4].astype(int)
        bboxes[:, 4] = bboxes[:, 4] % 1 + 0.001
        return bboxes, idx

    def _update_extra_features(self, obj, features, idx):
        obj.key_landmarks = features['landmarks'][idx]
        return obj




