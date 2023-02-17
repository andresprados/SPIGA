import copy

# Demo libs
import spiga.demo.analyze.extract.processor as pr


class VideoAnalyzer:
    def __init__(self, tracker, processor=pr.EmptyProcessor()):
        self.tracker = tracker
        self.processor = processor
        self.tracked_obj = []

    def process_frame(self, image):
        image = copy.copy(image)
        self.tracked_obj = self.tracker.process_frame(image, self.tracked_obj)
        if len(self.tracked_obj) > 0:
            self.tracked_obj = self.processor.process_frame(image, self.tracked_obj)
        self.tracked_obj = self._add_attributes()
        return self.tracked_obj

    def plot_features(self, image, plotter, show_attributes):
        for obj in self.tracked_obj:
            image = obj.plot_features(image, plotter, show_attributes)
        return image

    def get_attributes(self, names):

        # Check input type
        single_name = False
        if isinstance(names, str):
            names = [names]
            single_name = True

        attributes = {}
        for name in names:
            attribute = []
            for obj in self.tracked_obj:
                attribute.append(obj.get_attributes(name))
            attributes[name] = attribute

        if single_name:
            return attribute
        else:
            return attributes

    def _add_attributes(self):
        for obj in self.tracked_obj:
            if not obj.has_processor():
                obj.attributes += self.processor.attributes
                obj.attributes += self.tracker.attributes
                obj.drawers.append(self.processor.plot_features)
                obj.drawers.append(self.tracker.plot_features)
        return self.tracked_obj
