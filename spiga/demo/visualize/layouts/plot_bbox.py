import cv2

# Demo libs
from spiga.demo.visualize.layouts.plot_basics import BasicLayout


class BboxLayout(BasicLayout):

    BasicLayout.thickness_dft['bbox'] = 2

    def __init__(self):
        super().__init__()

    def draw_bbox(self, canvas, bbox, score_thr=0, show_score=True, thick=None, color=BasicLayout.colors['blue']):

        if thick is None:
            thick = self.thickness['bbox']

        if bbox[4] > score_thr:
            text = "{:.4f}".format(bbox[4])
            b = list(map(int, bbox))
            cv2.rectangle(canvas, (b[0], b[1]), (b[2], b[3]), color, thick)
            if show_score:
                self.draw_bbox_text(canvas, b, text, offset=(0, 12), color=color)
        return canvas

    def draw_bbox_line(self, canvas, bbox, score_thr=0, show_score=True, thick=None, color=BasicLayout.colors['blue']):

        if thick is None:
            thick = self.thickness['bbox']

        if bbox[4] > score_thr:
            text = "{:.4f}".format(bbox[4])
            b = list(map(int, bbox))
            cv2.line(canvas, (b[0], b[1]), (b[0], b[1] + 15), color, thick)
            cv2.line(canvas, (b[0], b[1]), (b[0] + 100, b[1]), color, thick)
            if show_score:
                self.draw_bbox_text(canvas, b, text, offset=(0, 12), color=color)
        return canvas

    def draw_bbox_text(self, canvas, bbox, text, offset=(0, 0), color=BasicLayout.colors['white']):
        b = list(map(int, bbox))
        cx = b[0] + offset[0]
        cy = b[1] + offset[1]
        cv2.putText(canvas, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, color)
        return canvas

    def draw_bboxes(self, canvas, dets, score_thr=0, show_score=True, thick=None, colors=(BasicLayout.colors['blue'])):
        num_colors = len(colors)
        for idx, bbox in enumerate(dets):
            color = colors[idx % num_colors]
            canvas = self.draw_bbox(canvas, bbox, score_thr=score_thr, show_score=show_score, thick=thick, color=color)
        return canvas
