
class Tracker:
    """
     Object detection and tracking interface in a video stream
    """
    def __init__(self):
        self.attributes = []

    def process_frame(self, image, tracked_obj):
        """
        Detect and track objects in the input image.
        :param image: OpenCV image.
        :param tracked_obj: List with the objects found.
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
