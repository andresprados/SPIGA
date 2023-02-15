import os
import numpy as np
import json
from collections import OrderedDict
from scipy.integrate import simps

from spiga.data.loaders.dl_config import db_anns_path
from spiga.eval.benchmark.metrics.metrics import Metrics


class MetricsLandmarks(Metrics):

    def __init__(self, name='landmarks'):
        super().__init__(name)

        self.db_info = None
        self.nme_norm = "corners"
        self.nme_thr = 8
        self.percentile = [90, 95, 99]
        # Cumulative plot axis length
        self.bins = 10000

    def compute_error(self, data_anns, data_pred, database, select_ids=None):

        # Initialize global logs and variables of Computer Error function
        self.init_ce(data_anns, data_pred, database)
        self._update_lnd_param()

        # Order data and compute nme
        self.error['nme_per_img'] = []
        self.error['ne_per_img'] = OrderedDict()
        self.error['ne_per_ldm'] = OrderedDict()
        for img_id, anns in enumerate(data_anns):
            # Init variables per img
            pred = data_pred[img_id]

            # Get select ids to compute
            if select_ids is None:
                selected_ldm = anns['ids']
            else:
                selected_ldm = list(set(select_ids) & set(anns['ids']))

            norm = self._get_img_norm(anns)
            for ldm_id in selected_ldm:
                # Compute Normalize Error
                anns_ldm = self._get_lnd_from_id(anns, ldm_id)
                pred_ldm = self._get_lnd_from_id(pred, ldm_id)
                ne = self._dist_l2(anns_ldm, pred_ldm)/norm * 100
                self.error['ne_per_img'].setdefault(img_id, []).append(ne)
                self.error['ne_per_ldm'].setdefault(ldm_id, []).append(ne)

            # NME per image
            if self.database in ['merlrav']:
                # LUVLI at MERLRAV divide by 68 despite the annotated landmarks in the image.
                self.error['nme_per_img'].append(np.sum(self.error['ne_per_img'][img_id])/68)
            else:
                self.error['nme_per_img'].append(np.mean(self.error['ne_per_img'][img_id]))

        # Cumulative NME
        self.error['cumulative_nme'] = self._cumulative_error(self.error['nme_per_img'], bins=self.bins)

        return self.error

    def metrics(self):

        # Initialize global logs and variables of Metrics function
        self.init_metrics()

        # Basic metrics (NME/NMPE/AUC/FR) for full dataset
        nme, nmpe, auc, fr, _, _ = self._basic_metrics()

        print('NME: %.3f' % nme)
        self.metrics_log['nme'] = nme
        for percent_id, percentile in enumerate(self.percentile):
            print('NME_P%i: %.3f' % (percentile, nmpe[percent_id]))
            self.metrics_log['nme_p%i' % percentile] = nmpe[percent_id]
        self.metrics_log['nme_thr'] = self.nme_thr
        self.metrics_log['nme_norm'] = self.nme_norm
        print('AUC_%i: %.3f' % (self.nme_thr, auc))
        self.metrics_log['auc'] = auc
        print('FR_%i: %.3f' % (self.nme_thr, fr))
        self.metrics_log['fr'] = fr

        # Subset basic metrics
        subsets = self.db_info['test_subsets']
        if self.data_type == 'test' and len(subsets) > 0:
            self.metrics_log['subset'] = OrderedDict()
            for subset, img_filter in subsets.items():
                self.metrics_log['subset'][subset] = OrderedDict()
                nme, nmpe, auc, fr, _, _ = self._basic_metrics(img_select=img_filter)
                print('> Landmarks subset: %s' % subset.upper())
                print('NME: %.3f' % nme)
                self.metrics_log['subset'][subset]['nme'] = nme
                for percent_id, percentile in enumerate(self.percentile):
                    print('NME_P%i: %.3f' % (percentile, nmpe[percent_id]))
                    self.metrics_log['subset'][subset]['nme_p%i' % percentile] = nmpe[percent_id]
                print('AUC_%i: %.3f' % (self.nme_thr, auc))
                self.metrics_log['subset'][subset]['auc'] = auc
                print('FR_%i: %.3f' % (self.nme_thr, fr))
                self.metrics_log['subset'][subset]['fr'] = fr

        # NME/NPE per landmark
        self.metrics_log['nme_per_ldm'] = OrderedDict()
        for percentile in self.percentile:
            self.metrics_log['npe%i_per_ldm' % percentile] = OrderedDict()
        for k, v in self.error['ne_per_ldm'].items():
            self.metrics_log['nme_per_ldm'][k] = np.mean(v)
            for percentile in self.percentile:
                self.metrics_log['npe%i_per_ldm' % percentile][k] = np.percentile(v, percentile)

        return self.metrics_log

    def get_pimg_err(self, data_dict=None, img_select=None):
        data = self.error['nme_per_img']
        if img_select is not None:
            data = [data[img_id] for img_id in img_select]
        name_dict = self.name + '/nme'
        if data_dict is not None:
            data_dict[name_dict] = data
        else:
            data_dict = data
        return data_dict

    def _update_lnd_param(self):
        db_info_file = db_anns_path.format(database=self.database, file_name='db_info')
        if os.path.exists(db_info_file):
            with open(db_info_file) as jsonfile:
                self.db_info = json.load(jsonfile)

            norm_dict = self.db_info['norm']
            nme_norm, nme_thr = next(iter(norm_dict.items()))
            print('Default landmarks configuration: \n %s: %i' % (nme_norm, nme_thr))
            answer = input("Change default config? (N/Y) >>> ")
            if answer.lower() in ['yes', 'y']:
                answer = input("Normalization options: %s >>> " % str(list(norm_dict.keys())))
                if answer in norm_dict.keys():
                    nme_norm = answer
                    nme_thr = norm_dict[nme_norm]
                else:
                    print("Option %s not available keep in default one: %s" % (answer, nme_norm))
                answer = input("Change threshold ->%s:%i ? (N/Y) >>> " % (nme_norm, nme_thr))
                if answer.lower() in ['yes', 'y']:
                    answer = input('NME threshold: >>> ')
                    nme_thr = float(answer)
                else:
                    print("Keeping default threshold: %i" % nme_thr)

            self.nme_norm = nme_norm
            self.nme_thr = nme_thr

        else:
            raise ValueError('Database %s specifics not defined. Missing db_info.json' % self.database)

    def _dist_l2(self, pointA, pointB):
        return float(((pointA - pointB) ** 2).sum() ** 0.5)

    def _get_lnd_from_id(self, anns, ids):
        idx = anns['ids'].index(ids)
        ref = np.array(anns['landmarks'][idx])
        return ref

    def _get_img_norm(self, anns):
        if self.nme_norm == 'pupils':
            print('WARNING: Pupils norm only implemented for 68 landmark configuration')
            left_eye = [7, 138, 139, 8, 141, 142]
            right_eye = [11, 144, 145, 12, 147, 148]
            refA = np.zeros(2)
            refB = np.zeros(2)
            for i in range(len(left_eye)):
                refA += self._get_lnd_from_id(anns, left_eye[i])
                refB += self._get_lnd_from_id(anns, right_eye[i])
            refA = refA/len(left_eye)   # Left
            refB = refB/len(right_eye)  # Right
        elif self.nme_norm == 'corners':
            refA = self._get_lnd_from_id(anns, 12)  # Left
            refB = self._get_lnd_from_id(anns, 7)  # Right
        elif self.nme_norm == 'diagonal':
            refA = anns['bbox'][0:2]
            refB = refA + anns['bbox'][2:4]
        elif self.nme_norm == 'height':
            return anns['bbox'][3]
        elif self.nme_norm == 'lnd_bbox':
            lnd = np.array(anns['landmarks'])
            lnd_max = np.max(lnd, axis=0)
            lnd_min = np.min(lnd, axis=0)
            lnd_wh = lnd_max - lnd_min
            return (lnd_wh[0]*lnd_wh[1])**0.5
        elif self.nme_norm == 'bbox':
            return (anns['bbox'][2] * anns['bbox'][3]) ** 0.5
        else:
            raise ValueError('Normalization %s not implemented' % self.nme_norm)

        return self._dist_l2(refA, refB)

    def _cumulative_error(self, error, bins=10000):
        num_imgs, base = np.histogram(error, bins=bins)
        cumulative = [x / float(len(error)) for x in np.cumsum(num_imgs)]
        base = base[:bins]
        cumulative, base = self._filter_cumulative(cumulative, base)
        return [cumulative, base]

    def _filter_cumulative(self, cumulative, base):
        base = [x for x in base if (x < self.nme_thr)]
        cumulative = cumulative[:len(base)]
        return cumulative, base

    def _basic_metrics(self, img_select=None):
        data = self.error['nme_per_img']
        if img_select is not None:
            data = [data[img_id] for img_id in img_select]
            [cumulative, base] = self._cumulative_error(data, bins=self.bins)
        else:
            [cumulative, base] = self.error['cumulative_nme']

        # Normalize Mean Error across img
        nme = np.mean(data)
        # Normalize Mean Percentile Error across img
        nmpe = []
        for percentile in self.percentile:
            nmpe.append(np.percentile(data, percentile))

        # Area Under Curve and Failure Rate
        auc, fr = self._auc_fr_metrics(cumulative, base)

        return nme, nmpe, auc, fr, cumulative, base

    def _auc_fr_metrics(self, cumulative, base):
        if not base:
            auc = 0.
            fr = 100.
        else:
            auc = (simps(cumulative, x=base) / self.nme_thr) * 100.0
            if base[-1] < self.nme_thr and cumulative[-1] == 1:
                auc += ((self.nme_thr - base[-1]) / self.nme_thr) * 100
            fr = (1 - cumulative[-1]) * 100.0
        return auc, fr
