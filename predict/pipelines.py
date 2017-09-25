
"""Frameworks for applying machine learning algorithms to -omics datasets.

This file contains classes used to organize feature selection, normalization,
and prediction methods into robust pipelines that can be used to infer
phenotypic information from -omic datasets.

See Also:
    :module:`../features/cohorts`: Storing -omic and phenotypic data.
    :module:`.classifiers`: Specific algorithms for binary prediction.
    :module:`.regressors`: Specific algorithms for continuous prediction.

Author: Michal Grzadkowski <grzadkow@ohsu.edu>

"""

from .cross_validation import (OmicRandomizedCV, cross_val_predict_omic,
                               OmicShuffleSplit)

import numpy as np
from abc import abstractmethod
import inspect

from numbers import Number
from functools import reduce
from operator import mul

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

    def _fit(self, X, y=None, **fit_params):
        self._validate_steps()
        step_names = [name for name, _ in self.steps]

        fit_params_steps = {name: {} for name, step in self.steps
                            if step is not None}

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
        return self.predict(omic_data)

    @staticmethod
    def parse_preds(preds):
        return preds

    def predict_omic(self, omic_data):
        """Gets a vector of phenotype predictions for an -omic dataset."""
        return self.parse_preds(self.predict_base(omic_data))

    @classmethod
    def extra_fit_params(cls, cohort):
        fit_params = {}

        if hasattr(cohort, 'path'):
            fit_params.update({'path_obj': cohort.path})

        return fit_params

    @classmethod
    def extra_tune_params(cls, cohort):
        return cls.extra_fit_params(cohort)

    @staticmethod
    def score_pheno(X, y):
        """The function used to calculate the accuracy of phenotype values
           predicted by the pipeline and the actual values themselves."""
        raise PipelineError("Class `OmicPipe` can't be used for prediction, "
                            "use its subclasses instead!")

    def score(self, X, y=None):
        """Get the accuracy of the classifier in predicting phenotype values.
        
        Used to ensure compatibility with cross-validation methods
        implemented in :module:`sklearn`.

        Args:
            X (array-like), shape = [n_samples, n_genes]
                A matrix of -omic values.

            y (array-like), shape = [n_samples, ]
                A vector of phenotype values.

        Returns:
            S (float): A score corresponding to prediction accuracy.
                The way this score is calculated is determined by the
                pipeline's `score_pheno` method.

        """
        raise PipelineError("Class `OmicPipe` can't be used for prediction, "
                            "use its subclasses instead!")


    def tune_coh(self,
                 cohort, pheno,
                 tune_splits=2, test_count=8, parallel_jobs=16,
                 include_samps=None, exclude_samps=None,
                 include_genes=None, exclude_genes=None,
                 verbose=False):
        """Tunes the pipeline by sampling over the tuning parameters."""

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

        # checks if the classifier has parameters to be tuned, and how many
        # parameter combinations are possible
        if self.tune_priors:
            prior_counts = [len(x) if hasattr(x, '__len__') else float('Inf')
                            for x in self.cur_tuning.values()]
            max_tests = reduce(mul, prior_counts, 1)
            test_count = min(test_count, max_tests)

            # samples parameter combinations and tests each one
            grid_test = OmicRandomizedCV(
                estimator=self, param_distributions=self.cur_tuning,
                n_iter=test_count, cv=tune_cvs, refit=False,
                n_jobs=parallel_jobs, pre_dispatch='n_jobs'
                )
            grid_test.fit(X=train_omics, y=train_pheno,
                          **self.extra_tune_params(cohort))

            # finds the best parameter combination and updates the classifier
            tune_scores = (grid_test.cv_results_['mean_test_score']
                           - grid_test.cv_results_['std_test_score'])
            self.set_params(
                **grid_test.cv_results_['params'][tune_scores.argmax()])

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

        return self.fit(X=train_omics, y=train_pheno,
                        **self.extra_fit_params(cohort))

    def score_coh(self,
                  cohort, pheno,
                  score_splits=16,
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
        omics = cohort.train_omics(include_samps, exclude_samps,
                                   include_genes, exclude_genes)
        pheno_types = cohort.train_pheno(pheno, omics.index)

        score_cvs = OmicShuffleSplit(
            n_splits=score_splits, test_size=0.2,
            random_state=cohort.intern_cv_)

        return np.percentile(
            cross_val_score(estimator=self,
                            X=cohort.train_omics, y=pheno_types,
                            fit_params=self.extra_fit_params(cohort),
                            cv=score_cvs, n_jobs=-1),
            25
            )

    def eval_coh(self,
                 cohort, pheno,
                 include_samps=None, exclude_samps=None,
                 include_genes=None, exclude_genes=None):
        """Evaluate the performance of a classifier."""

        test_omics, test_pheno = cohort.test_data(
            pheno,
            include_samps, exclude_samps,
            include_genes, exclude_genes
            )

        if len(test_pheno) < 5:
            return float('nan')

        else:
            return self.score(test_omics, test_pheno)

    def infer_coh(self,
                  cohort, pheno,
                  infer_splits=16,
                  include_samps=None, exclude_samps=None,
                  include_genes=None, exclude_genes=None):
        omics = cohort.train_omics(include_samps, exclude_samps,
                                   include_genes, exclude_genes)
        pheno_types = cohort.train_pheno(pheno, omics.index)

        return cross_val_predict_mut(
            estimator=self,
            X=omics, y=pheno_types,
            cv_fold=5, cv_count=infer_splits,
            fit_params=self.extra_fit_params(cohort),
            random_state=int(cohort.intern_cv_ ** 1.5) % 42949672, n_jobs=-1
            )

    def get_coef(self):
        """Get the fitted coefficient for each gene in the -omic dataset."""

        if self.genes is None:
            raise PipelineError("Gene coefficients only available once "
                                "the pipeline has been fit!")

        else:
            return {gn: 0 for gn in self.genes}


class UniPipe(OmicPipe):
    """Predicting a single phenotype from -omics dataset(s).
    
    """

    def fit(self, X, y=None, **fit_params):
        """Fits the steps of the pipeline in turn."""

        Xt, final_params = self._fit(
            X, y.ravel(),
            **{**fit_params,
               **{'expr_genes': [xcol.split('__')[-1]
                                 for xcol in X.columns]}}
            )

        if 'feat' in self.named_steps:
            self.genes = X.columns[
                self.named_steps['feat']._get_support_mask()]
        else:
            self.genes = X.columns

        if 'genes' in final_params:
            final_params['genes'] = self.genes
        if self._final_estimator is not None:
            self._final_estimator.fit(Xt, y.ravel(), **final_params)

        return self
    
    def score(self, X, y=None):
        return self.score_pheno(np.array(y).flatten(),
                                np.array(self.predict_omic(X)).flatten())


class LinearPipe(UniPipe):
    """A class corresponding to linear logistic regression classification
       of mutation status. For classes implementing specific
       regularization penalties.
    """

    def get_coef(self):
        return {gene: coef for gene, coef in
                zip(self.genes, self.named_steps['fit'].coef_[0])}


class TransferPipe(OmicPipe):
    """A pipeline that transfers information between multiple datasets.

    """

    def fit(self, X, y=None, **fit_params):
        """Fits the steps of the pipeline in turn."""

        self.prot_genes = y.columns

        Xt_list, final_params = self._fit(
            X, y,
            **{**fit_params, **{
                'expr_genes': {
                    lbl: [xcol.split('__')[-1] for xcol in Xmat.columns]
                    for lbl, Xmat in X.items()
                    }}}
            )

        if 'feat' in self.named_steps:
            sup_mask = self.named_steps['feat']._get_support_mask()
            self.genes = {lbl: Xmat.columns[sup_mask[lbl]]
                          for lbl, Xmat in X.items()}

        else:
            self.genes = {lbl: Xmat.columns for lbl, Xmat in X.items()}

        if 'genes' in final_params:
            final_params['genes'] = self.genes

        if self._final_estimator is not None:
            self._final_estimator.fit(
                Xt_list, y, **{'prot_genes': self.prot_genes,
                               **final_params}
                )

        return self


class MultiPipe(OmicPipe):
    """A pipeline for predicting multiple phenotypes at once.

    """

    def predict_omic(self, omic_data):
        return [self.parse_preds(preds)
                for preds in self.predict_base(omic_data)]

    @staticmethod
    def parse_scores(scores):
        """Summarizes the scores across the predicted variates."""
        return np.min(scores)

    @abstractmethod
    def score_each(self, omics, pheno):
        """Scores each of the given phenotypes separately."""

    def score_pheno(self, X, y):
        return self.parse_scores([self.score_each(omics, pheno)
                                  for omics, pheno in zip(X, y)])


class ParallelPipe(TransferPipe):
    """A pipeline for predicting phenotypes in multiple cohorts at once.

    """

    def tune_coh(self,
                 cohorts, pheno,
                 tune_splits=2, test_count=8, parallel_jobs=16,
                 include_samps=None, exclude_samps=None,
                 include_genes=None, exclude_genes=None,
                 verbose=False):
        """Tunes the pipeline by sampling over the tuning parameters."""

        train_data = [cohort.train_data(pheno,
                                        include_samps, exclude_samps,
                                        include_genes, exclude_genes)
                      for cohort in cohorts]

        # get internal cross-validation splits in the training set and use
        # them to tune the classifier
        tune_cvs = OmicShuffleSplit(
            n_splits=tune_splits, test_size=0.2,
            random_state=(cohort.cv_seed ** 2) % 42949672
            )

        # checks if the classifier has parameters to be tuned, and how many
        # parameter combinations are possible
        if self.tune_priors:
            prior_counts = [len(x) if hasattr(x, '__len__') else float('Inf')
                            for x in self.cur_tuning.values()]
            max_tests = reduce(mul, prior_counts, 1)
            test_count = min(test_count, max_tests)

            # samples parameter combinations and tests each one
            grid_test = OmicRandomizedCV(
                estimator=self, param_distributions=self.cur_tuning,
                n_iter=test_count, cv=tune_cvs, refit=False,
                n_jobs=parallel_jobs, pre_dispatch='n_jobs'
                )
            grid_test.fit(X=[omics for omics, _ in train_data],
                          y=[pheno for _, pheno in train_data],
                          **self.extra_tune_params(cohort))

            # finds the best parameter combination and updates the classifier
            tune_scores = (grid_test.cv_results_['mean_test_score']
                           - grid_test.cv_results_['std_test_score'])
            self.set_params(
                **grid_test.cv_results_['params'][tune_scores.argmax()])

            if verbose:
                print(self)

        return self

    def fit_coh(self,
                cohorts, pheno,
                include_samps=None, exclude_samps=None,
                include_genes=None, exclude_genes=None):
        """Fits a classifier."""

        train_data = [cohort.train_data(pheno,
                                        include_samps, exclude_samps,
                                        include_genes, exclude_genes)
                      for cohort in cohorts]

        return self.fit(X=[omics for omics, _ in train_data],
                        y=[pheno for _, pheno in train_data],
                        **self.extra_fit_params(cohort))

    def eval_coh(self,
                 cohorts, pheno,
                 include_samps=None, exclude_samps=None,
                 include_genes=None, exclude_genes=None):
        """Evaluate the performance of a classifier."""

        test_data = [cohort.test_data(pheno,
                                      include_samps, exclude_samps,
                                      include_genes, exclude_genes)
                     for cohort in cohorts]

        if all([len(pheno) < 5 for _, pheno in test_data]):
            return float('nan')

        else:
            return self.score([omics for omics, _ in test_data],
                              [pheno for _, pheno in test_data])


class PresencePipe(OmicPipe):
    """A class corresponding to pipelines which use continuous data to
       predict discrete outcomes.
    """

    def __init__(self, steps, path_keys=None):
        if not hasattr(steps[-1][-1], 'predict_proba'):
            raise PipelineError(
                "Variant pipelines must have a classification estimator"
                "with a 'predict_proba' method as their final step!"
                )

        super().__init__(steps, path_keys)

    def parse_preds(self, preds):
        if hasattr(self, 'classes_'):
            true_indx = [i for i, x in enumerate(self.classes_) if x]
            parse_preds = [scrs[true_indx] for scrs in preds]

        else:
            parse_preds = preds

        return parse_preds

    def predict_base(self, omic_data):
        return self.predict_proba(omic_data)

    @staticmethod
    def score_pheno(X, y):
        if len(np.unique(X)) == 1:
            return 0.5
        else:
            return roc_auc_score(X, y)


class ValuePipe(OmicPipe):
    """A class corresponding to pipelines which use continuous data to
       predict continuous outcomes.
    """
    pass


class ProteinPipe(ValuePipe):
    """A class corresponding to pipelines for predicting proteomic levels."""
    
    @staticmethod
    def score_pheno(X, y):
        return pearsonr(X, y)[0]


class DrugPipe(ValuePipe):

    @staticmethod
    def score_pheno(X, y):
        return r2_score(X, y, multioutput='variance_weighted')


class VariantPipe(PresencePipe):
    """A class corresponding to pipelines for predicting discrete gene
       mutation states such as SNPs, indels, and frameshifts.
    """

    def __repr__(self):
        """Prints the classifier name and the feature selection path key."""
        return '{}_{}'.format(
            type(self).__name__, str(self.get_params()['path_keys']))


class MutPipe(UniPipe, VariantPipe):
    """A class corresponding to pipelines for predicting
       discrete gene mutation states individually.
    """

    @classmethod
    def extra_fit_params(cls, cohort):
        return {'mut_genes': cohort.mut_genes,
                'path_obj': cohort.path}




class MultiPresencePipe(MultiPipe, PresencePipe):
    """A class corresponding to pipelines for predicting a collection of
       discrete gene mutation states simultaenously.
    """

    def predict_mut(self, expr):
        """Returns the probability of mutation presence calculated by the
           classifier based on the given expression matrix.
        """
        return self.predict_proba(expr)

    def predict_labels(self, expr):
        expr_t = self.transform(expr)
        return self.predict_labels(expr_t)

    def score_coh(self,
                  cohort, pheno,
                  score_splits=16,
                  include_samples=None, exclude_samples=None,
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
        samps, genes = cohort._validate_dims(mtype=mtype,
                                             exclude_samps=exclude_samps,
                                             gene_list=gene_list)

        score_cvs = OmicShuffleSplit(
            n_splits=score_splits, test_size=0.2,
            random_state=cohort.intern_cv_)

        return np.percentile(cross_val_score(
            estimator=self,
            X=cohort.train_expr_.loc[samps, genes],
            y=cohort.train_mut_.status(samps, mtype),
            fit_params={'feat__mut_genes': cohort.mut_genes,
                        'feat__path_obj': cohort.path,
                        'fit__mut_genes': cohort.mut_genes,
                        'fit__path_obj': cohort.path},
            scoring=self.score_mut, cv=score_cvs, n_jobs=-1
            ), 25)

    def infer_coh(self,
                  cohort, pheno,
                  infer_splits=16,
                  include_samples=None, exclude_samples=None,
                  include_genes=None, exclude_genes=None):
        samps, genes = cohort._validate_dims(gene_list=gene_list)

        infer_scores = cross_val_predict_mut(
            estimator=self,
            X=cohort.train_expr_.loc[:, genes],
            y=cohort.train_mut_.status(samps, mtype),
            exclude_samps=exclude_samps, cv_fold=4, cv_count=infer_splits,
            fit_params={'feat__mut_genes': cohort.mut_genes,
                        'feat__path_obj': cohort.path,
                        'fit__mut_genes': cohort.mut_genes,
                        'fit__path_obj': cohort.path},
            random_state=int(cohort.intern_cv_ ** 1.5) % 42949672, n_jobs=-1
            )

        return infer_scores

