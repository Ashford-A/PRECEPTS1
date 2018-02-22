
from ..pipelines import MutPipe
from ..selection import *

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler

import numpy as np
from scipy import stats
import pystan


class BaseStan(BaseEstimator, ClassifierMixin):

    model_name = 'BasePredict'

    def __init__(self, model_code):

        self.model_code = model_code
        self.expr_genes = None
        self.data_dict = None

        self.fit_obj = None
        self.summary = None

    @staticmethod
    def get_data_dict(X, y, **fit_params):
        raise NotImplementedError(
            "Class must implement method get_data_dict!")

    def run_model(self, X, y, verbose=False, **fit_params):
        raise NotImplementedError("Class must implement method run_model!")

    def get_var_means(self):
        raise NotImplementedError(
            "Class must implement method get_var_means!")

    @staticmethod
    def calc_pred_p(X, var_means):
        raise NotImplementedError("Class must implement method calc_pred_p!")

    def fit(self, X, y=None, expr_genes=None, verbose=False, **fit_params):
        self.X = X
        self.y = y
        self.expr_genes = expr_genes

        self.stan_model = pystan.StanModel(
            model_code=self.model_code, model_name=self.model_name,
            verbose=verbose
            )

        self.data_dict = self.get_data_dict(X, y, **fit_params)
        self.run_model(X, y, verbose, **fit_params)

        return self

    def predict_proba(self, X, **fit_params):
        if self.fit_obj is None:
            raise ValueError("Model has not been fit yet!")

        return self.calc_pred_p(X, self.get_var_means())

    def get_params(self, deep=True):
        return {'model_code': self.model_code}


class LogitStan(BaseStan):

    model_name = 'LogisticPredict'

    @staticmethod
    def get_data_dict(X, y, **fit_params):
        return {'N': X.shape[0], 'G': X.shape[1], 'expr': X, 'mut': y + 0}

    @staticmethod
    def calc_pred_p(X, var_means):
        return np.dot(X, var_means['gn_wghts']) + var_means['alpha']


class MarginStan(BaseStan):

    model_name = 'MarginalPredict'

    def __init__(self,
                 model_code, wt_distr=(-1.0, 0.5), mut_distr=(1.0, 0.5)):
        self.wt_distr = wt_distr
        self.mut_distr = mut_distr

        super().__init__(model_code)

    def get_data_dict(self, X, y, **fit_params):
        return {'Nw': np.sum(~y), 'Nm': np.sum(y), 'G': X.shape[1],
                'expr_w': X[~y], 'expr_m': X[y],
                'wt_distr': self.wt_distr, 'mut_distr': self.mut_distr}

    def calc_pred_p(self, X, var_means):
        pred_scores = np.dot(X, var_means['gn_wghts']) + var_means['alpha']

        neg_logl = stats.norm.logpdf(pred_scores, *self.wt_distr)
        pos_logl = stats.norm.logpdf(pred_scores, *self.mut_distr)
        pred_p = 1 / (1 + np.exp(np.clip(neg_logl - pos_logl, -100, 100)))

        return pred_scores


class StanOptimizing(BaseStan):

    def get_var_means(self):
        var_means = {}

        for var, vals in self.fit_obj.items():
            if vals.shape:
                var_means[var] = vals
            else:
                var_means[var] = float(vals)

        return var_means

    def run_model(self, X, y=None, verbose=False, **fit_params):
        self.fit_obj = self.stan_model.optimizing(
            data=self.data_dict, verbose=verbose, iter=1e5)

        if verbose:
            print("Fitting has finished!")


class StanVariational(BaseStan):

    def get_var_means(self):
        var_means = {}

        for var, val in zip(self.fit_obj['sampler_param_names'],
                            self.fit_obj['mean_pars']):
            var_parse = var.split('.')

            if len(var_parse) == 1:
                var_means[var_parse[0]] = val

            else:
                if var_parse[0] in var_means:
                    var_means[var_parse[0]] += [val]
                else:
                    var_means[var_parse[0]] = [val]

        return var_means

    def run_model(self, X, y=None, verbose=True, **fit_params):
        self.fit_obj = self.stan_model.vb(
            data=self.data_dict, iter=2e4, verbose=verbose)

        if verbose:
            print("Fitting has finished!")


class StanSampling(BaseStan):

    def get_var_means(self):
        var_means = {} 

        for var, val in zip(self.fit_obj.flatnames, self.summary[:, 0]):
            var_parse = var.split('[')

            if len(var_parse) == 1:
                var_means[var_parse[0]] = val

            else:
                if var_parse[0] in var_means:
                    var_means[var_parse[0]] += [val]
                else:
                    var_means[var_parse[0]] = [val]

        return var_means

    def run_model(self,
                  X, y=None, expr_genes=None, verbose=True, **fit_params):

        if 'n_chains' not in fit_params:
            n_chains = 1
        else:
            n_chains = fit_params['n_chains']

        self.fit_obj = self.stan_model.sampling(
            data=self.data_dict, chains=n_chains, iter=50, n_jobs=n_chains,
            verbose=verbose
            )

        if verbose:
            print("Fitting has finished!")

        self.summary = dict(self.fit_obj.summary())['summary']


class StanPipe(MutPipe):

    def __init__(self, fit_inst, path_keys=None):
        feat_step = PathwaySelect(path_keys=path_keys)
        norm_step = StandardScaler()
        fit_step = fit_inst
        self.fit_inst = fit_inst

        super().__init__(
            [('feat', feat_step), ('norm', norm_step), ('fit', fit_step)],
            path_keys=path_keys
            )

    def get_params(self, deep=True):
        param_dict = super().get_params(deep)
        param_dict.update({'fit_inst': self.fit_inst})

        return param_dict

    @classmethod
    def extra_fit_params(cls, cohort):
        return {**super().extra_fit_params(cohort), **{'n_chains': 8}}

    @classmethod
    def extra_tune_params(cls, cohort):
        return {**super().extra_tune_params(cohort), **{'n_chains': 2}}

    def get_coef(self):
        pass

