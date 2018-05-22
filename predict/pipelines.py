
"""Frameworks for applying machine learning algorithms to -omics datasets.

This file contains classes used to organize feature selection, normalization,
and prediction methods into robust pipelines that can be used to infer
phenotypic information from -omic datasets.

See Also:
    :module:`../features/cohorts`: Storing -omic and phenotypic data.
    :module:`.basic/classifiers`: Specific algorithms for binary prediction.

Author: Michal Grzadkowski <grzadkow@ohsu.edu>

"""

from .cross_validation import (OmicRandomizedCV, cross_val_predict_omic,
                               OmicShuffleSplit)

import numpy as np

from numbers import Number
from functools import reduce
from operator import mul
from copy import copy

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

from sklearn.metrics import roc_auc_score, r2_score
from scipy.stats import pearsonr


class PipelineError(Exception):
    pass


class OmicPipe(Pipeline):
    """Extracting phenotypic predictions from -omics dataset(s).

    Args:
        steps (list): A series of transformations and classifiers.
            An ordered list of feature selection, normalization, and
            classification/regression steps, the last of which produces
            feature predictions.

    """

    # the parameters that are to be tuned, with either statistical
    # distributions or iterables to be sampled from as values
    tune_priors = {}

    def __init__(self, steps, path_keys=None):
        super().__init__(steps)
        self.genes = None
        self.cur_tuning = dict(self.tune_priors)
        self.path_keys = path_keys

        self.tune_params_add = None
        self.fit_params_add = None

    def __str__(self):
        """Prints the tuned parameters of the pipeline."""
        param_str = type(self).__name__ + ' with '

        if self.tune_priors:
            param_list = self.get_params()
            param_str += reduce(
                lambda x, y: x + ', ' + y,
                [k + ': ' + '%s' % float('%.4g' % param_list[k])
                 if isinstance(param_list[k], Number)
                 else k + ': ' + str(param_list[k])
                 for k in self.cur_tuning.keys()]
                )
        else:
            param_str += 'no tuned parameters.'

        return param_str

    def fit(self, X, y=None, **fit_params):
        """Fits the steps of the pipeline in turn."""

        Xt, final_params = self._fit(
            X, y,
            **{**fit_params, **{
                'expr_genes': [
                    xcol.split('__')[-1] if isinstance(xcol, str)
                    else xcol[0].split('__')[-1] for xcol in X.columns
                    ],
                'expr_cols': X.columns
                }}
            )

        if 'feat' in self.named_steps:
            self.genes = X.columns[
                self.named_steps['feat']._get_support_mask()]

        else:
            self.genes = X.columns

        if 'genes' in final_params:
            final_params['genes'] = self.genes
        if self._final_estimator is not None:
            self._final_estimator.fit(Xt, y, **final_params)

        return self

    def _fit(self, X, y=None, **fit_params):
        self._validate_steps()
        step_names = [name for name, _ in self.steps]

        fit_params_steps = {name: {} for name, step in self.steps
                            if step is not None}

        if 'fit' in fit_params_steps and self.fit_params_add:
            for pname, pval in self.fit_params_add.items():
                fit_params_steps['fit'][pname] = pval

        for pname, pval in fit_params.items():
            if '__' in pname:
                step, param = pname.split('__', maxsplit=1)
                fit_params_steps[step][param] = pval

            else:
                for step in fit_params_steps:
                    step_indx = step_names.index(step)

                    if (pname in self.steps[step_indx][1].fit
                            .__code__.co_varnames):
                        fit_params_steps[step][pname] = pval

        Xt = X

        for name, transform in self.steps[:-1]:

            if transform is None:
                pass

            elif hasattr(transform, "fit_transform"):
                Xt = transform.fit_transform(Xt, y, **fit_params_steps[name])

            else:
                Xt = transform.fit(Xt, y, **fit_params_steps[name]) \
                    .transform(Xt)

        if self._final_estimator is None:
            final_params = {}
        else:
            final_params = fit_params_steps[self.steps[-1][0]]

        return Xt, final_params

    def predict_train(self,
                      cohort, pheno,
                      include_samps=None, exclude_samps=None,
                      include_genes=None, exclude_genes=None):
        return self.predict_omic(
            cohort.train_data(pheno,
                              include_samps, exclude_samps,
                              include_genes, exclude_genes)[0]
            )

    def predict_test(self,
                     cohort, pheno,
                     include_samps=None, exclude_samps=None,
                     include_genes=None, exclude_genes=None):
        return self.predict_omic(
            cohort.test_data(pheno,
                             include_samps, exclude_samps,
                             include_genes, exclude_genes)[0]
            )

    def predict_base(self, omic_data):
        """Applies the prediction method specific to the pipeline."""
        return self.predict(omic_data)

    @staticmethod
    def parse_preds(preds):
        return np.array(preds).flatten()

    def predict_omic(self, omic_data):
        """Gets a vector of phenotype predictions for an -omic dataset."""
        return self.parse_preds(self.predict_base(omic_data))

    @classmethod
    def extra_fit_params(cls, cohort):
        fit_params = {}

        #if hasattr(cohort, 'path'):
        #    fit_params.update({'path_obj': cohort.path})

        return fit_params

    @classmethod
    def extra_tune_params(cls, cohort):
        return cls.extra_fit_params(cohort)

    def score(self, X, y=None, sample_weight=None):
        """Get the accuracy of the classifier in predicting phenotype values.
        
        Used to ensure compatibility with cross-validation methods
        implemented in :module:`sklearn`.

        Args:
            X (array-like), shape = [n_samples, n_genes]
                A matrix of -omic values.

            y (array-like), shape = [n_samples, ]
                A vector of phenotype values.

            sample_weight (array-like), default = None
                If not None, how much weight to assign to the prediction for
                each sample when calculating the score.

        Returns:
            S (float): A score corresponding to prediction accuracy.
                The way this score is calculated is determined by the
                pipeline's `score_pheno` method.

        """
        return self.score_omic(y, self.predict_omic(X))

    def score_omic(self, actual_omic, pred_omic):
        """Parses and scores the predictions for a set of phenotypes."""
        return self.score_pheno(actual_omic.flatten(), pred_omic)

    @staticmethod
    def score_pheno(actual_pheno, pred_pheno):
        """Scores the predicted values for a single phenotype."""
        raise NotImplementedError("An -omic pipeline used for prediction "
                                  "must implement the <score_pheno> method!")

    def tune_coh(self,
                 cohort, pheno,
                 tune_splits=2, test_count=8, parallel_jobs=16,
                 include_samps=None, exclude_samps=None,
                 include_genes=None, exclude_genes=None,
                 verbose=False):
        """Tunes the pipeline by sampling over the tuning parameters."""

        # checks if the classifier has parameters to be tuned, and how many
        # parameter combinations are possible
        if self.tune_priors:
            prior_counts = [len(x) if hasattr(x, '__len__') else float('Inf')
                            for x in self.cur_tuning.values()]
            max_tests = reduce(mul, prior_counts, 1)
            test_count = min(test_count, max_tests)

            train_omics, train_pheno = cohort.train_data(
                pheno,
                include_samps, exclude_samps,
                include_genes, exclude_genes
                )

            # get internal cross-validation splits in the training set and use
            # them to tune the classifier
            tune_cvs = OmicShuffleSplit(
                n_splits=tune_splits, test_size=0.2,
                random_state=(cohort.cv_seed ** 2) % 42949672
                )

            # samples parameter combinations and tests each one
            grid_test = OmicRandomizedCV(
                estimator=self, param_distributions=self.cur_tuning,
                n_iter=test_count, cv=tune_cvs, refit=False,
                n_jobs=parallel_jobs, pre_dispatch='n_jobs'
                )

            #TODO: figure out why passing extra_tune_params breaks in the new
            # scikit-learn code
            extra_params = self.extra_tune_params(cohort)
            grid_test.fit(X=train_omics, y=train_pheno, **extra_params)

            # finds the best parameter combination and updates the classifier
            tune_scores = (grid_test.cv_results_['mean_test_score']
                           - grid_test.cv_results_['std_test_score'])
            best_indx = tune_scores.argmax()

            best_params = grid_test.cv_results_['params'][best_indx]
            for par in best_params.keys() & extra_params.keys():
                del best_params[par]

            self.set_params(**best_params)
            if verbose:
                print(self)

        return self

    def fit_coh(self,
                cohort, pheno,
                include_samps=None, exclude_samps=None,
                include_genes=None, exclude_genes=None):
        """Fits a classifier."""

        train_omics, train_pheno = cohort.train_data(
            pheno,
            include_samps, exclude_samps,
            include_genes, exclude_genes
            )
        self.fit_params_add = self.extra_fit_params(cohort)

        return self.fit(X=train_omics, y=train_pheno)

    def fit_transform_coh(self,
                          cohort, pheno=None,
                          include_samps=None, exclude_samps=None,
                          include_genes=None, exclude_genes=None):

        train_omics, train_pheno = cohort.train_data(
            pheno,
            include_samps, exclude_samps,
            include_genes, exclude_genes
            )
        self.fit_params_add = self.extra_fit_params(cohort)

        return self.fit_transform(X=train_omics, y=train_pheno)

    def score_coh(self,
                  cohort, pheno, score_splits=16, parallel_jobs=8,
                  include_samps=None, exclude_samps=None,
                  include_genes=None, exclude_genes=None):
        """Test a classifier using tuning and cross-validation
           within the training samples of this dataset.

        Parameters
        ----------
        clf : UniClassifier
            An instance of the classifier to test.

        mtype : MuType, optional
            The mutation sub-type to test the classifier on.
            Default is to use all of the mutations available.

        verbose : boolean
            Whether or not the classifier should print information about the
            optimal hyper-parameters found during tuning.

        Returns
        -------
        P : float
            The 1st quartile of tuned classifier performance across the
            cross-validation samples. Used instead of the mean of performance
            to take into account performance variation for "hard" samples.

            Performance is measured using the area under the receiver operator
            curve metric.

        """

        train_omics, train_pheno = cohort.train_data(
            pheno,
            include_samps, exclude_samps,
            include_genes, exclude_genes
            )

        # get internal cross-validation splits in the training set and use
        # them to score the performance of the classifier
        score_cvs = OmicShuffleSplit(
            n_splits=score_splits, test_size=0.2,
            random_state=(cohort.cv_seed ** 3) % 42949672
            )

        return np.percentile(
            cross_val_score(estimator=self,
                            X=train_omics, y=train_pheno,
                            fit_params=self.extra_tune_params(cohort),
                            cv=score_cvs, n_jobs=parallel_jobs),
            25
            )

    def eval_coh(self,
                 cohort, pheno, use_train=False,
                 include_samps=None, exclude_samps=None,
                 include_genes=None, exclude_genes=None):
        """Evaluate the performance of a classifier."""

        if use_train:
            test_omics, test_pheno = cohort.train_data(
                pheno,
                include_samps, exclude_samps,
                include_genes, exclude_genes
                )

        else:
            test_omics, test_pheno = cohort.test_data(
                pheno,
                include_samps, exclude_samps,
                include_genes, exclude_genes
                )

        return self.score(test_omics, test_pheno)

    def tune_fit_eval(self,
                      cohort, pheno,
                      tune_splits=2, test_count=8, parallel_jobs=16,
                      include_samps=None, exclude_samps=None,
                      include_genes=None, exclude_genes=None):

        self.tune_coh(cohort, pheno, tune_splits, test_count, parallel_jobs,
                      include_samps, exclude_samps,
                      include_genes, exclude_genes)

        self.fit_coh(cohort, pheno,
                     include_samps, exclude_samps,
                     include_genes, exclude_genes)

        eval_score = self.eval_coh(cohort, pheno,
                                   include_samps, exclude_samps,
                                   include_genes, exclude_genes)

        return eval_score

    def infer_coh(self,
                  cohort, pheno, force_test_samps=None,
                  infer_splits=16, infer_folds=4, parallel_jobs=8,
                  include_samps=None, exclude_samps=None,
                  include_genes=None, exclude_genes=None):

        train_omics, train_pheno = cohort.train_data(
            pheno,
            include_samps, exclude_samps,
            include_genes, exclude_genes
            )

        return cross_val_predict_omic(
            estimator=self, X=train_omics, y=train_pheno,
            force_test_samps=force_test_samps,
            cv_fold=infer_folds, cv_count=infer_splits, n_jobs=parallel_jobs,
            fit_params=self.extra_fit_params(cohort),
            random_state=int(cohort.cv_seed ** 1.5) % 42949672,
            )

    def get_coef(self):
        """Get the fitted coefficient for each gene in the -omic dataset."""

        if self.genes is None:
            raise PipelineError("Gene coefficients only available once "
                                "the pipeline has been fit!")

        else:
            return {gn: 0 for gn in self.genes}


class PresencePipe(OmicPipe):
    """A class corresponding to pipelines which use continuous data to
       predict discrete outcomes.
    """

    def __init__(self, steps, path_keys=None):
        if not (hasattr(steps[-1][-1], 'predict_proba')
                or 'predict_proba' in steps[-1][-1].__class__.__dict__):

            raise PipelineError(
                "Variant pipelines must have a classification estimator "
                "with a 'predict_proba' method as their final step!"
                )

        super().__init__(steps, path_keys)

    def parse_preds(self, preds):
        if hasattr(self, 'classes_'):
            true_indx = [i for i, x in enumerate(self.classes_) if x]

            if len(true_indx) < 1:
                raise PipelineError("Classifier doesn't have a <True> class!")

            elif len(true_indx) > 1:
                raise PipelineError("Classifier has multiple <True> classes!")

            parse_preds = [scrs[true_indx[0]] for scrs in preds]

        else:
            parse_preds = preds

        return parse_preds

    def predict_base(self, omic_data):
        return self.predict_proba(omic_data)

    @staticmethod
    def score_pheno(actual_pheno, pred_pheno):
        pheno_score = 0.5

        if (len(np.unique(actual_pheno)) > 1
                and len(np.unique(pred_pheno)) > 1):
            pheno_score = roc_auc_score(actual_pheno, pred_pheno)

        return pheno_score


class ValuePipe(OmicPipe):
    """A class corresponding to pipelines which use continuous data to
       predict continuous outcomes.
    """

    @staticmethod
    def score_pheno(actual_pheno, pred_pheno):
        if np.var(actual_pheno) == 0 or np.var(pred_pheno) == 0:
            return 0
        else:
            return pearsonr(actual_pheno, pred_pheno)[0]


class TransferPipe(OmicPipe):
    """A pipeline that transfers information between multiple datasets.

    """

    def _fit(self, X_dict, y_dict=None, **fit_params):
        if y_dict is None:
            y_dict = {lbl: None for lbl in X_dict}

        self._validate_steps()
        step_names = [name for name, _ in self.steps]

        fit_params_steps = {name: {} for name, step in self.steps
                            if step is not None}

        if 'fit' in fit_params_steps and self.fit_params_add:
            for pname, pval in self.fit_params_add.items():
                fit_params_steps['fit'][pname] = pval

        for pname, pval in fit_params.items():
            if '__' in pname:
                step, param = pname.split('__', maxsplit=1)
                fit_params_steps[step][param] = pval

            else:
                for step in fit_params_steps:
                    step_indx = step_names.index(step)

                    if (pname in self.steps[step_indx][1].fit
                            .__code__.co_varnames):
                        fit_params_steps[step][pname] = pval

        self.lbl_transforms = {lbl: [] for lbl in X_dict}
        Xt_dict = {lbl: None for lbl in X_dict}

        for lbl in X_dict:
            Xt = X_dict[lbl]
            
            for name, transform in self.steps[:-1]:
                if transform is None:
                    pass
                
                elif hasattr(transform, "fit_transform"):
                    Xt = transform.fit_transform(
                        Xt, y_dict[lbl], **fit_params_steps[name])

                else:
                    Xt = transform.fit(
                        Xt, y_dict[lbl],
                        **fit_params_steps[name]
                        ).transform(Xt)

                self.lbl_transforms[lbl] += [(name, copy(transform))]

            Xt_dict[lbl] = Xt

        if self._final_estimator is None:
            final_params = {}
        else:
            final_params = fit_params_steps[self.steps[-1][0]]

        return Xt_dict, final_params

    def fit(self, X, y=None, **fit_params):
        """Fits the steps of the pipeline in turn."""

        expr_genes = {lbl: [xcol.split('__')[-1] for xcol in X_mat.columns]
                      for lbl, X_mat in X.items()}
        Xt_dict, final_params = self._fit(
            X, y, **{**fit_params, **{'expr_genes': expr_genes}})

        if 'feat' in self.named_steps:
            self.genes = {
                lbl: X_mat.columns[
                    self.named_steps['feat']._get_support_mask()]
                for lbl, X_mat in X.items()
                }

        else:
            self.genes = {lbl: X_mat.columns for lbl, X_mat in X.items()}

        if 'genes' in final_params:
            final_params['genes'] = self.genes

        if self._final_estimator is not None:
            self._final_estimator.fit(Xt_dict, y, **final_params)

        return self

    def predict_proba(self, X):
        Xt_dict = X

        for lbl in X:
            for name, transform in self.lbl_transforms[lbl]:

                if transform is not None:
                    Xt_dict[lbl] = transform.transform(Xt_dict[lbl])

        return self.steps[-1][-1].predict_proba(Xt_dict)

    def eval_coh_each(self,
                      cohort, pheno, use_train=False,
                      include_samps=None, exclude_samps=None,
                      include_genes=None, exclude_genes=None):

        if use_train:
            test_omics, test_pheno = cohort.train_data(
                pheno,
                include_samps, exclude_samps,
                include_genes, exclude_genes
                )

        else:
            test_omics, test_pheno = cohort.test_data(
                pheno,
                include_samps, exclude_samps,
                include_genes, exclude_genes
                )

        return self.score_cohorts(test_omics, test_pheno)

    @staticmethod
    def parse_preds(preds):
        return {lbl: np.array(x).flatten() for lbl, x in preds.items()}

    def score_omic(self, actual_omic, pred_omic):
        """Parses and scores the predictions for a set of phenotypes."""
        return np.min(tuple(self.score_each(actual_omic, pred_omic).values()))

    def score_cohorts(self, X, y=None, sample_weight=None):
        return self.score_each(y, self.predict_omic(X))

    def score_each(self, actual_omic, pred_omic):
        return {lbl: self.score_pheno(actual_omic[lbl].flatten(),
                                      pred_omic[lbl])
                for lbl in actual_omic}


class MultiPipe(OmicPipe):
    """A pipeline for predicting multiple phenotypes at once.

    """

    @staticmethod
    def parse_preds(preds):
        return np.array(preds)

    def score_omic(self, actual_omic, pred_omic):
        """
        Args:
            actual_omic, pred_omic: np.array, shape: (n_samps, n_phenos)
        """
        return np.min(self.score_each(actual_omic, pred_omic))

    def score_each(self, actual_omic, pred_omic):
        return [self.score_pheno(act_omic, p_omic)
                for act_omic, p_omic in zip(actual_omic.transpose(),
                                            pred_omic.transpose())]


class ProteinPipe(ValuePipe):
    """A class corresponding to pipelines for predicting proteomic levels."""

    @staticmethod
    def score_pheno(actual_pheno, pred_pheno):
        if np.var(actual_pheno) == 0 or np.var(pred_pheno) == 0:
            return 0
        else:
            return r2_score(actual_pheno, pred_pheno)


class MutPipe(PresencePipe):
    """A class corresponding to pipelines for predicting
       discrete gene mutation states individually.
    """

    def __repr__(self):
        """Prints the classifier name and the feature selection path key."""
        return '{}_{}'.format(
            type(self).__name__, str(self.get_params()['path_keys']))


class LinearPipe(OmicPipe):
    """An abstract class for classifiers implementing a linear separator.

    """

    def get_coef(self):
        return {gene: coef for gene, coef in
                zip(self.genes, self.named_steps['fit'].coef_.flatten())}


class EnsemblePipe(OmicPipe):
    """An abstract class for classifiers made up of ensembles of separators.

    """

    def fit(self, X, y=None, **fit_params):
        self.effect_direct = [
            ((X.iloc[y.flatten(), i].mean() - X.iloc[~y.flatten(), i].mean())
             > 0)
            for i in range(X.shape[1])
            ]

        return super().fit(X, y, **fit_params)

    def get_coef(self):
        return {gene: coef * (2 * direct - 1) for gene, coef, direct in
                zip(self.genes, self.named_steps['fit'].feature_importances_,
                    self.effect_direct)}

