import pkg_resources
import json
import copy
import torch

import spiga.data.loaders.dl_config as dl_cfg
import spiga.data.loaders.dataloader as dl
import spiga.inference.pretreatment as pretreat
from spiga.inference.framework import SPIGAFramework
from spiga.inference.config import ModelConfig


def main():
    import argparse
    pars = argparse.ArgumentParser(description='Experiment results generator')
    pars.add_argument('database', type=str, help='Database name',
                      choices=['wflw', '300wpublic', '300wprivate', "merlrav", "cofw68"])
    pars.add_argument('-a','--anns', type=str, default='test', help='Annotations type: test, valid or train')
    pars.add_argument('--gpus', type=int, default=0, help='GPU Id')
    args = pars.parse_args()

    # Load model framework
    model_cfg = ModelConfig(args.database)
    model_framework = SPIGAFramework(model_cfg, gpus=[args.gpus])

    # Generate results
    tester = Tester(model_framework, args.database, anns_type=args.anns)
    with torch.no_grad():
        tester.generate_results()


class Tester:

    def __init__(self, model_framework, database, anns_type='test'):

        # Parameters
        self.anns_type = anns_type
        self.database = database

        # Model initialization
        self.model_framework = model_framework

        # Dataloader
        self.dl_eval = dl_cfg.AlignConfig(self.database, mode=self.anns_type)
        self.dl_eval.aug_names = []
        self.dl_eval.shuffle = False
        self.dl_eval.target_dist = self.model_framework.model_cfg.target_dist
        self.dl_eval.image_size = self.model_framework.model_cfg.image_size
        self.dl_eval.ftmap_size = self.model_framework.model_cfg.ftmap_size

        self.batch_size = 1
        self.test_data, _ = dl.get_dataloader(self.batch_size, self.dl_eval,
                                              pretreat=pretreat.NormalizeAndPermute(), debug=True)

        # Results
        self.data_struc = {'imgpath': str, 'bbox': None, 'headpose': None, 'ids': None, 'landmarks': None, 'visible': None}
        self.result_path = pkg_resources.resource_filename('spiga', 'eval/results')
        self.result_file = '/results_%s_%s.json' % (self.database, self.anns_type)
        self.file_out = self.result_path + self.result_file

    def generate_results(self):

        data = []
        for step, batch in enumerate(self.test_data):
            print('Step: ', step)
            inputs = self.model_framework.select_inputs(batch)
            outputs_raw = self.model_framework.net_forward(inputs)
            # Postprocessing
            outputs = self.model_framework.postreatment(outputs_raw, batch['bbox'], batch['bbox_raw'])

            # Data
            data_dict = copy.deepcopy(self.data_struc)
            data_dict['imgpath'] = batch['imgpath_local'][0]
            data_dict['bbox'] = batch['bbox_raw'][0].numpy().tolist()
            data_dict['visible'] = batch['visible'][0].numpy().tolist()
            data_dict['ids'] = self.dl_eval.database.ldm_ids
            data_dict['landmarks'] = outputs['landmarks'][0]
            data_dict['headpose'] = outputs['headpose'][0]
            data.append(data_dict)

        # Save outputs
        with open(self.file_out, 'w') as outfile:
            json.dump(data, outfile)


if __name__ == '__main__':
    main()
