
from ..base import *

import numpy as np

from sklearn.metrics import pairwise
from scipy.stats import norm
from scipy.special import expit


class OrthoTransfer(StanClassifier):

    model_name = "OrthoMultiTaskTransfer"

    def __init__(self, model_code, weights):
        self.weights = weights
        super().__init__(model_code)


class GeneModel(OrthoTransfer):

    model_name = 'GeneBasedOrthoMultiTaskTransfer'

    def __init__(self,
                 model_code, weights=((1.0, 1.0), (1.0, -1.0)), alpha=0.005):
        self.alpha = alpha
        super().__init__(model_code, weights)

    def get_data_dict(self, omic, pheno, **fit_params):
        return {'N': omic.shape[0], 'G': omic.shape[1],
                'T': pheno.shape[1], 'R': len(self.weights),
                'expr': omic, 'muts': np.array(pheno + 0),
                'weights': np.array(self.weights), 'alpha': self.alpha}

    def calc_pred_p(self, omic, var_means):
        return expit(
            np.dot(np.dot(omic, var_means['proj_vals']),
                   np.array(self.weights))
            + var_means['intercept']
            )


class MarginOrthoTransfer(OrthoTransfer):

    model_name = 'MarginBasedOrthoMultiTaskTransfer'

    def __init__(self,
                 model_code, weights=((1.0, 1.0), (1.0, 0.0), (0.0, 1.0)),
                 wt_distr=(-1.0, 1.0), mut1_distr=(1.0, 1.0),
                 mut2_distr=(1.0, 1.0), alpha=0.01):
        self.wt_distr = wt_distr
        self.mut1_distr = mut1_distr
        self.mut2_distr = mut2_distr
        self.alpha = alpha

        super().__init__(model_code, weights)

    def get_data_dict(self, omic, pheno, **fit_params):
        wt_stat = ~np.any(pheno, axis=1)

        return {'Nw': np.sum(wt_stat), 'Nmi': np.sum(pheno[:, 0]),
                'Nmj': np.sum(pheno[:, 1]), 'G': omic.shape[1],
                'expr_w': omic[wt_stat], 'expr_mi': omic[pheno[:, 0]],
                'expr_mj': omic[pheno[:, 1]], 'R': len(self.weights),
                'weights': np.array(self.weights), 'wt_distr': self.wt_distr,
                'mut1_distr': self.mut1_distr, 'mut2_distr': self.mut2_distr,
                'alpha': self.alpha}

    def calc_pred_labels(self, omic):
        var_means = self.get_var_means()
        hidden_feats = np.dot(omic, var_means['proj_vals'])

        return np.dot(hidden_feats,
                      np.array(self.weights)) + var_means['intercept']

    def calc_pred_p(self, pred_labels):
        neg_logl = norm.logpdf(pred_labels, *self.wt_distr)
        pos1_logl = norm.logpdf(pred_labels[:, 0], *self.mut1_distr)
        pos2_logl = norm.logpdf(pred_labels[:, 1], *self.mut2_distr)

        return np.stack(
            [1 / (1 + np.exp(np.clip(neg_logl[:, 0] - pos1_logl, -100, 100))),
             1 / (1 + np.exp(np.clip(neg_logl[:, 1] - pos2_logl, -100, 100)))],
            axis=1
            )

