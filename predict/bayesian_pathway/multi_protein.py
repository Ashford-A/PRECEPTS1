
from ..pipelines import MultiPipe, TransferPipe, ValuePipe
from ..selection import IntxTypeSelect
from .stan_models import *

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
            path_obj=None, n_chains=8, parallel_jobs=24,
            verbose=False, **fit_params):

        # ensures given input data is in the correct format
        if not isinstance(X, dict):
            raise TypeError("X must be a dictionary!")

        if 'rna' not in X:
            raise ValueError("X must have a `rna` entry!")

        if 'cna' not in X:
            raise ValueError("X must have a `cna` entry!")

        # gets the list of genes we can use in the model
        both_genes = sorted(list(
            set(X['rna'].columns) & set(X['cna'].columns)
            & set(fit_params['prot_genes'])
            ))

        # gets the pathway interactions we can use in our model
        use_path = [(up_gn, down_gn) for up_gn, down_gn in
                    path_obj[self.path_type]
                    if up_gn in both_genes and down_gn in both_genes]

        if verbose:
            print("\nConsidering {} interactions between {} genes.\n".format(
                        len(use_path), len(both_genes)))

        x_rna = X['rna'].loc[:, both_genes]
        x_cna = X['cna'].loc[:, both_genes]
        y_use = y.loc[:, both_genes]

        # constructs the pathway edges between each gene and itself
        # to feed into the model
        path_out = [x + 1 for x in range(len(both_genes))]
        path_in = [x + 1 for x in range(len(both_genes))]

        # adds the edges defined by the given pathway interactions
        path_out += [x_rna.columns.get_loc(up_gn) + 1
                     for up_gn, down_gn in use_path]
        path_in += [x_rna.columns.get_loc(down_gn) + 1
                    for up_gn, down_gn in use_path]

        # provides initial values to use for edge weights in the model
        path_wght = [0.8 for _ in both_genes] + [0.05 for _ in use_path]

        # initializes the Stan model and compiles it to C++ code
        sm = pystan.StanModel(model_code=model_code,
                              model_name='ProteinPredict', verbose=True)

        # lists the known data we will feed into the model
        data_dict = {'N': x_rna.shape[0], 'G': x_rna.shape[1],
                     'r': x_rna, 'c': x_cna, 'p': np.nan_to_num(y_use),
                     'P': len(path_out), 'po': path_out, 'pi': path_in}

        # fits the model given known data, initial values, and priors
        self.fit_obj = sm.sampling(
            chains=n_chains, n_jobs=parallel_jobs, iter=16,
            data=data_dict, init=[{'wght': path_wght} for _ in range(n_chains)],
            verbose=True
            )

        if verbose:
            print("Fitting has finished!")

        self.use_genes = both_genes
        self.use_path = use_path

    def predict(self, X, verbose=False, **fit_params):

        if self.fit_obj is None:
            raise ValueError("Model has not been fit yet!")

        if verbose:
            print("Creating predictions...")

        # reorders the given RNA and CNA input to match the data
        # the model was trained on
        x_rna = X['rna'].loc[:, self.use_genes]
        x_cna = X['cna'].loc[:, self.use_genes]

        # extract more detailed information about sampled parameters, current
        # PyStan implementation (v2.17) makes this too slow however:

        # print('Extracting combination estimates...')
        # comb_data = self.fit_obj.summary(pars='comb')['summary'][:, 0]
        # print('Extracting edge weight estimates...')
        # path_wghts = self.fit_obj.summary(pars='wght')['summary'][:, 0]

        # get the names of the variables in the model and their posterior
        # means, find the chain with the best log-posterior
        var_names = self.fit_obj.flatnames
        post_means = self.fit_obj.get_posterior_mean()
        best_chain = post_means[-1, :].argmax()

        # gets the fitted values of the rna-cna combination weights
        tx_wghts = post_means[[i for i, nm in enumerate(var_names)
                               if 'tx_wght[' in nm],
                              best_chain]

        # gets the fitted values of the pathway edge weights
        edge_wghts = post_means[[i for i, nm in enumerate(var_names)
                                 if 'edge_wght[' in nm],
                                best_chain]

        tx_mat = (x_rna * tx_wghts) + (x_cna * (1 - tx_wghts))
        pred_p = tx_mat * edge_wghts[:len(self.use_genes)]

        for pt, wght in zip(self.use_path, edge_wghts[len(self.use_genes):]):
            pred_p[pt[1]] += tx_mat[pt[0]] * wght

        return pred_p


class StanProteinPredictEns(BaseEstimator, RegressorMixin):

    def __init__(self, path_type, precision=0.05, known_prots=None):
        self.path_type = path_type
        self.precision = precision
        self.known_prots = known_prots

        self.use_genes = None
        self.use_path = None

    def fit(self,
            X, y=None, path_obj=None, n_chains=8, parallel_jobs=24,
            verbose=True, **fit_params):

        # ensures given input data is in the correct format
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

        both_genes = [gn for gn in both_genes
                      if gn not in self.known_prots.columns]

        x_rna = X['rna'].loc[:, both_genes]
        x_cna = X['cna'].loc[:, both_genes]

        k_prots = self.known_prots.loc[x_rna.index, :]
        y_use = y.loc[:, both_genes]

        path_out = [x + 1 for x in range(len(both_genes))]
        path_in = [x + 1 for x in range(len(both_genes))]

        path_out += [x_rna.columns.get_loc(up_gn) + 1
                     for up_gn, down_gn in use_path
                     if up_gn in both_genes and down_gn in both_genes]
        path_in += [x_rna.columns.get_loc(down_gn) + 1
                    for up_gn, down_gn in use_path
                    if up_gn in both_genes and down_gn in both_genes]

        path_out += [k_prots.columns.get_loc(up_gn) + 1 + len(both_genes)
                     for up_gn, down_gn in use_path
                     if up_gn in k_prots.columns and down_gn in both_genes]
        path_in += [x_rna.columns.get_loc(down_gn) + 1
                    for up_gn, down_gn in use_path
                    if up_gn in k_prots.columns and down_gn in both_genes]

        if verbose:
            print("{} interactions found between {} genes".format(
                len(path_out), len(both_genes)))

        init_wghts = [{'wght': []} for _ in range(n_chains)]
        for i in range(n_chains):

            init_wghts[i]['wght'] += [round(0.5 + (i * 0.45 / n_chains), 2)
                                      for _ in both_genes]
            init_wghts[i]['wght'] += [
                round((0.45 / n_chains) * (n_chains - i), 2)
                for _ in range(len(path_out) - len(both_genes))
                ]

        sm = pystan.StanModel(model_code=model_code_ens,
                              verbose=True, model_name="ProteinPredict")

        self.fit_obj = sm.sampling(
            iter=10, chains=n_chains, n_jobs=parallel_jobs,
            data={'N': x_rna.shape[0], 'G': x_rna.shape[1],
                  'R': k_prots.shape[1], 'prec': self.precision,
                  'r': x_rna, 'c': x_cna, 'p': np.nan_to_num(y_use),
                  'k': k_prots, 'P': len(path_out), 'po': path_out,
                  'pi': path_in},
            init=init_wghts, verbose=True,
            )

        print("Fitting has finished!")

        self.use_genes = both_genes
        self.use_path = use_path

    def predict(self, X, **fit_params):

        print("Creating predictions...")
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

    @classmethod
    def extra_fit_params(cls, cohort):
        return {**super().extra_fit_params(cohort),
                **{'n_chains': 4, 'parallel_jobs': 4}}

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
            score_val = pearsonr(y_true[~nan_stat], y_pred[~nan_stat])[0]

        else:
            score_val = np.nan

        return score_val

    def score_pheno(self, y_true, y_pred):
        return self.parse_scores(
            [self.score_each(
                y_true[gn],
                y_pred[:, self.named_steps['fit'].use_genes.index(gn)]
                )
             for gn in y_true]
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


class StanDefault(StanProteinPipe):

    def __init__(self, intx_type=None):
        feat_step = IntxTypeSelect(path_keys=intx_type)
        fit_step = StanProteinPredict(path_type=intx_type)

        super().__init__([('feat', feat_step), ('fit', fit_step)])
        self.intx_type = intx_type


class StanEnsemble(StanProteinPipe):

    def __init__(self, intx_type=None, known_prots=None):
        feat_step = IntxTypeSelect(path_keys=intx_type)
        fit_step = StanProteinPredictEns(path_type=intx_type,
                                         known_prots=known_prots)

        super().__init__([('feat', feat_step), ('fit', fit_step)])
        self.intx_type = intx_type

