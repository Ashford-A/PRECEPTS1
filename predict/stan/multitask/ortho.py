
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

