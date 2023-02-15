from collections import OrderedDict


class Metrics:

    def __init__(self, name='metrics'):

        # Data dicts
        self.error = OrderedDict()
        self.metrics_log = OrderedDict()
        self.name = name
        self.database = None
        self.data_type = None

    def compute_error(self, data_anns, data_pred, database, select_ids=None):
        self.init_ce(data_anns, data_pred, database)
        raise ValueError('Computer error has to be implemented by inheritance')

    def init_ce(self, data_anns, data_pred, database):
        # Update database info
        [self.database, self.data_type] = database
        # Logs and checks
        print('Computing %s error...' % self.name)
        if len(data_anns) == 0:
            raise ValueError('Annotations miss for computing error in %s' % self.name)
        if len(data_pred) == 0:
            raise ValueError('Predictions miss for computing error in %s' % self.name)
        elif len(data_pred) != len(data_anns):
            raise Warning('Prediction vs annotations length mismatch')

    def metrics(self):
        self.init_metrics()
        raise ValueError('Metrics has to be implemented by inheritance')

    def init_metrics(self):
        # Logs and checks
        print('> Metrics %s:' % self.name)
        if len(self.error) == 0:
            raise ValueError('Error must be compute first in %s' % self.name)

    def get_pimg_err(self, data_dict):
        return data_dict
