
from ..pipelines import MultiPipe, TransferPipe, ValuePipe
from ..selection import IntxTypeSelect
from .stan_models import model_code

from scipy.stats import pearsonr
from sklearn.base import BaseEstimator, RegressorMixin

import pystan
import numpy as np


class StanProteinPredict(BaseEstimator, RegressorMixin):

    def __init__(self, path_type, precision=0.01):
        self.path_type = path_type
        self.precision = precision

        self.use_genes = None
        self.use_path = None

    def fit(self,
            X, y=None,
            path_obj=None, n_chains=8, parallel_jobs=8, **fit_params):
        if not isinstance(X, dict):
            raise TypeError("X must be a dictionary!")

        if 'rna' not in X:
            raise ValueError("X must have a `rna` entry!")

        if 'cna' not in X:
            raise ValueError("X must have a `cna` entry!")

        both_genes = sorted(list(
            set(X['rna'].columns) & set(X['cna'].columns)
            & set(fit_params['prot_genes'])
            ))

        use_path = [(up_gn, down_gn) for up_gn, down_gn in
                    path_obj[self.path_type]
                    if up_gn in both_genes and down_gn in both_genes]

        x_rna = X['rna'].loc[:, both_genes]
        x_cna = X['cna'].loc[:, both_genes]
        y_use = y.loc[:, both_genes]

        path_out = [x + 1 for x in range(len(both_genes))]
        path_in = [x + 1 for x in range(len(both_genes))]

        path_out += [x_rna.columns.get_loc(up_gn) + 1
                     for up_gn, down_gn in use_path]
        path_in += [x_rna.columns.get_loc(down_gn) + 1
                    for up_gn, down_gn in use_path]

        path_wght = [0.99 for _ in both_genes] + [0.05 for _ in use_path]

        self.fit_obj = pystan.stan(
            model_code=model_code,
            iter=20, chains=n_chains, n_jobs=parallel_jobs,
            data={'N': x_rna.shape[0], 'G': x_rna.shape[1],
                  'r': x_rna, 'c': x_cna, 'p': np.nan_to_num(y_use),
                  'P': len(path_out), 'po': path_out, 'pi': path_in},
            init=[{'wght': path_wght} for _ in range(n_chains)],
            model_name="ProteinPredict"
            )

        self.use_genes = both_genes
        self.use_path = use_path

    def predict(self, X, **fit_params):

        if self.fit_obj is None:
            raise ValueError("Model has not been fit yet!")

        x_rna = X['rna'].loc[:, self.use_genes]
        x_cna = X['cna'].loc[:, self.use_genes]

        comb_data = self.fit_obj.summary(pars='comb')['summary'][:, 0]
        path_wghts = self.fit_obj.summary(pars='wght')['summary'][:, 0]

        act_sum = (x_rna * comb_data) + (x_cna * (1 - comb_data))
        pred_p = act_sum * path_wghts[:len(self.use_genes)]

        for pt, wght in zip(self.use_path, path_wghts[len(self.use_genes):]):
            pred_p[pt[1]] += act_sum[pt[0]] * wght

        return pred_p


class StanProteinPipe(MultiPipe, TransferPipe, ValuePipe):

    tune_priors = (
        ('fit__precision', (0.001, 0.005, 0.01, 0.05)),
        )

    def __init__(self, intx_type=None):
        feat_step = IntxTypeSelect(path_keys=intx_type)
        fit_step = StanProteinPredict(path_type=intx_type)

        super().__init__([('feat', feat_step), ('fit', fit_step)])
        self.intx_type = intx_type

    @classmethod
    def extra_fit_params(cls, cohort):
        return {**super().extra_fit_params(cohort),
                **{'n_chains': 12, 'parallel_jobs': 12}}

    @classmethod
    def extra_tune_params(cls, cohort):
        return {**super().extra_tune_params(cohort),
                **{'n_chains': 4, 'parallel_jobs': 1}}

    def predict_omic(self, omic_data):
        return self.predict_base(omic_data)

    @staticmethod
    def parse_scores(scores):
        return np.mean(np.array(scores)[~np.isnan(scores)])

    def score_each(self, y_true, y_pred):
        nan_stat = np.isnan(y_true) | np.isnan(y_pred)

        if np.sum(nan_stat) < (len(y_true) - 2):
            score_val = pearsonr(
                y_true.loc[~nan_stat], y_pred.loc[~nan_stat])[0]

        else:
            score_val = np.nan

        return score_val

    def score_pheno(self, y_true, y_pred):
        return self.parse_scores(
            [self.score_each(y_true.loc[:, j], y_pred.loc[:, j])
             for j in set(y_true.columns) & set(y_pred.columns)]
            )

    def get_coef(self):
        gn_coef = {gn: {'rna': {}, 'cna': {}}
                   for gn in self.named_steps['fit'].use_genes}

        comb_data = {gn: comb for gn, comb in
                     zip(self.named_steps['fit'].use_genes,
                         self.named_steps['fit'].fit_obj.summary(
                             pars='comb')['summary'][:, 0])}

        path_wghts = self.named_steps['fit'].fit_obj.summary(
            pars='wght')['summary'][:, 0]

        for i, gn in enumerate(self.use_genes):
            gn_coef[gn]['rna'][gn] = comb_data[gn] * path_wghts[i]
            gn_coef[gn]['cna'][gn] = (1 - comb_data[gn]) * path_wghts[i]

        for (up_gn, down_gn), wght in zip(self.named_steps['fit'].use_path,
                                          path_wghts[len(comb_data):]):

            gn_coef[down_gn]['rna'][up_gn] = comb_data[up_gn] * wght
            gn_coef[down_gn]['cna'][up_gn] = (1 - comb_data[up_gn]) * wght

        return gn_coef

