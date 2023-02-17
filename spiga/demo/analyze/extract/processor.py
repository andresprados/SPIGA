

class Processor:
    def __init__(self):
        self.attributes = []

    def process_frame(self, frame, tracked_obj):
        """
        Process tracked objects to extract interesting features.
        :param frame: OpenCV image.
        :param tracked_obj: List with the objects to be processed.
        """
        raise NotImplementedError()

    def plot_features(self, image, features, plotter, show_attributes):
        """
        Visualize objects detected in the input image.
        :param image: OpenCV image.
        :param features: List of object features detect after processing the frame.
        :param plotter: Plotter interface.
        :param show_attributes: Selected object attributes to be displayed.
        """
        raise NotImplementedError()


class EmptyProcessor(Processor):
    def __init__(self):
        super().__init__()

    def process_frame(self, frame, tracked_obj):
        return tracked_obj

    def plot_features(self, image, features, plotter, show_attributes):
        return image


class ProcessorsGroup(Processor):
    def __init__(self):
        super().__init__()
        self.group = []

    def process_frame(self, frame, tracked_obj):
        for elem in self.group:
            tracked_obj = elem.process_frame(frame, tracked_obj)
        return tracked_obj

    def plot_features(self, image, features, plotter, show_attributes):
        for elem in self.group:
            image = elem.plot_features(image, features, plotter, show_attributes)
        return image

    def add_processor(self, processor):
        self.group.append(processor)
        self.attributes += processor.attributes

    def get_number_of_processors(self):
        return len(self.group)

