import json
import pkg_resources
from collections import OrderedDict

# Paths
data_path = pkg_resources.resource_filename('spiga', 'data/annotations')

def main():

    import argparse
    pars = argparse.ArgumentParser(description='Benchmark alignments evaluator')
    pars.add_argument('pred_file', nargs='+', type=str, help='Absolute path to the prediction json file (Multi file)')
    pars.add_argument('--eval', nargs='+', type=str, default=['lnd'],
                      choices=['lnd', 'pose'], help='Evaluation modes')
    pars.add_argument('-s', '--save', action='store_true', help='Save results')
    args = pars.parse_args()

    for pred_file in args.pred_file:
        benchmark = get_evaluator(pred_file, args.eval, args.save)
        benchmark.metrics()


class Evaluator:

    def __init__(self, data_file, evals=(), save=True, process_err=True):

        # Inputs
        self.data_file = data_file
        self.evals = evals
        self.save = save

        # Paths
        data_name = data_file.split('/')[-1]
        self.data_dir = data_file.split(data_name)[0]

        # Information from name
        data_name = data_name.split('.')[0]
        data_name = data_name.split('_')
        self.data_type = data_name[-1]
        self.database = data_name[-2]

        # Load predictions and annotations
        anns_file = data_path + '/%s/%s.json' % (self.database, self.data_type)
        self.anns = self.load_files(anns_file)
        self.pred = self.load_files(data_file)

        # Compute errors
        self.error = OrderedDict()
        self.error_pimg = OrderedDict()
        self.metrics_log = OrderedDict()
        if process_err:
            self.compute_error(self.anns, self.pred)

    def compute_error(self, anns, pred, select_ids=None):
        database_ref = [self.database, self.data_type]
        for eval in self.evals:
            self.error[eval.name] = eval.compute_error(anns, pred, database_ref, select_ids)
            self.error_pimg = eval.get_pimg_err(self.error_pimg)
        return self.error

    def metrics(self):
        for eval in self.evals:
            self.metrics_log[eval.name] = eval.metrics()

        if self.save:
            file_name = self.data_dir + '/metrics_%s_%s.txt' % (self.database, self.data_type)
            with open(file_name, 'w') as file:
                file.write(str(self))

        return self.metrics_log

    def load_files(self, input_file):
        with open(input_file) as jsonfile:
            data = json.load(jsonfile)
        return data

    def _dict2text(self, name, dictionary, num_tab=1):
        prev_tabs = '\t'*num_tab
        text = '%s {\n' % name
        for k, v in dictionary.items():
            if isinstance(v, OrderedDict) or isinstance(v, dict):
                text += '{}{}'.format(prev_tabs, self._dict2text(k, v, num_tab=num_tab+1))
            else:
                text += '{}{}: {}\n'.format(prev_tabs, k, v)
        text += (prev_tabs + '}\n')
        return text

    def __str__(self):
        state_dict = self.metrics_log
        text = self._dict2text('Metrics', state_dict)
        return text


def get_evaluator(pred_file, evaluate=('lnd', 'pose'), save=False, process_err=True):
    eval_list = []
    if "lnd" in evaluate:
        import spiga.eval.benchmark.metrics.landmarks as mlnd
        eval_list.append(mlnd.MetricsLandmarks())
    if "pose" in evaluate:
        import spiga.eval.benchmark.metrics.pose as mpose
        eval_list.append(mpose.MetricsHeadpose())

    return Evaluator(pred_file, evals=eval_list, save=save, process_err=process_err)


if __name__ == '__main__':
    main()

