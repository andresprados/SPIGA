
class ObjectAnalyzed:

    def __init__(self):
        # Processor addons
        self.attributes = []
        self.drawers = []

    def has_processor(self):
        if len(self.attributes) > 0:
            return True
        else:
            return False

    def plot_features(self, image, plotter, show_attributes):
        for drawer in self.drawers:
            image = drawer(image, self, plotter, show_attributes)
        return image

    def get_attributes(self, names=None):

        # Initialization by input type
        single_name = False
        if names is None:
            names = self.attributes
        elif isinstance(names, str):
            names = [names]
            single_name = True

        attributes = {}
        attribute = []
        for name in names:
            if name in self.attributes and name in self.__dict__.keys():
                attribute = getattr(self, name)
                attributes[name] = attribute

        if single_name:
            return attribute
        else:
            return attributes