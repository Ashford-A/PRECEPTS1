
"""Frameworks for applying machine learning algorithms to -omics datasets.

This file contains classes used to organize feature selection, normalization,
and prediction methods into robust pipelines that can be used to infer
phenotypic information from -omic datasets.

See Also:
    :module:`../features/cohorts`: Storing -omic and phenotypic data.
    :module:`.classifiers`: Specific machine learning algorithms.

Author: Michal Grzadkowski <grzadkow@ohsu.edu>

"""

from .cross_validation import (
    MutRandomizedCV, cross_val_predict_mut, MutShuffleSplit, DrugShuffleSplit)

from abc import abstractmethod
import numpy as np
import inspect

from numbers import Number
from functools import reduce
from operator import mul

from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit


class OmicPipe(Pipeline):
    """Extracting phenotypic predictions from an -omics dataset.

    Args:
        steps (list): A series of transformations and classifiers.
            An ordered list of feature selection, normalization, and
            classification/regression steps, the last of which produces
            feature predictions.

    """

    # the parameters that are to be tuned, with either statistical
    # distributions or iterables to be sampled from as values
    tune_priors = {}

    # the method to use for randomly splitting the training cohort during
    # cross-validation, the default is to ensure that each split preserves
    # the distribution of class labels of the training cohort as a whole
    cvSplitMethod = StratifiedShuffleSplit

    def __init__(self, steps):
        super().__init__(steps)
        self.genes = None
        self.cur_tuning = dict(self.tune_priors)

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

    def predict_train(self,
                      cohort,
                      include_samps=None, exclude_samps=None,
                      include_genes=None, exclude_genes=None):
        return self.predict_omic(
            cohort.train_omics(include_samps, exclude_samps,
                               include_genes, exclude_genes)
            )

    def predict_test(self,
                     cohort,
                     include_samps=None, exclude_samps=None,
                     include_genes=None, exclude_genes=None):
        return self.predict_omic(
            cohort.test_omics(include_samps, exclude_samps,
                              include_genes, exclude_genes)
            )

    def predict_base(self, omic_data):
        return self.predict(omic_data)

    @classmethod
    def parse_preds(cls, preds):
        return preds

    def predict_omic(self, omic_data):
        """Gets a vector of phenotype predictions for an -omic dataset."""
        return self.parse_preds(self.predict_base(omic_data))

    @classmethod
    def extra_fit_params(cls, cohort):
        return {}

    def tune_coh(self,
                 cohort, pheno,
                 tune_splits=2, test_count=16, parallel_jobs=16,
                 include_samps=None, exclude_samps=None,
                 include_genes=None, exclude_genes=None,
                 verbose=False):
        """Tunes the pipeline by sampling over the tuning parameters."""

        omics = cohort.train_omics(include_samps, exclude_samps,
                                   include_genes, exclude_genes)
        pheno_types = cohort.train_pheno(pheno, omics.index)

        # get internal cross-validation splits in the training set and use
        # them to tune the classifier
        tune_cvs = self.cvSplitMethod(
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
            grid_test = RandomizedSearchCV(
                estimator=self, param_distributions=self.cur_tuning,
                fit_params=self.extra_fit_params(cohort),
                n_iter=test_count, cv=tune_cvs, refit=False,
                n_jobs=parallel_jobs, pre_dispatch='n_jobs'
                )
            grid_test.fit(omics, pheno_types)

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
        omics = cohort.train_omics(include_samps, exclude_samps,
                                   include_genes, exclude_genes)
        pheno_types = cohort.train_pheno(pheno, omics.index)

        return self.fit(X=omics, y=pheno_types,
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

        score_cvs = self.cvSplitMethod(
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
        omics = cohort.test_omics(include_samps, exclude_samps,
                                  include_genes, exclude_genes)
        pheno_types = cohort.test_pheno(pheno, omics.index)

        return self.score(omics, pheno_types)

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

    # is the 0 just a placeholder here? how does this work with the
    # abstract method with the same name?
    # is this complete?
    def get_coef(self):
        """Get the fitted coefficient for each gene in the -omic dataset."""

        if self.genes is None:
            return ValueError("Gene coefficients only available once the"
                              "pipeline has been fit!")
        else:
            return {gn: 0 for gn in self.genes}


class UniPipe(OmicPipe):

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

    def fit(self, X, y=None, **fit_params):
        """Fits the steps of the pipeline in turn."""

        Xt, final_params = self._fit(
            X, y, **{**fit_params, **{'expr_genes': X.columns}})

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


class MultiPipe(OmicPipe):

    def predict_omic(self, omic_data):
        return [self.parse_preds(preds)
                for preds in self.predict_base(omic_data)]

    @classmethod
    def parse_scores(cls, scores):
        """Summarizes the scores across the predicted variates."""
        return np.min(scores)

    @abstractmethod
    def score_each(self, X, y=None):
        """Scores each of the given phenotypes separately."""

    def score(self, X, y=None):
        return self.parse_scores(self.score_each(X, y))


class LabelPipe(OmicPipe):
    """A class corresponding to pipelines which use continuous data to
       predict discrete outcomes.
    """

    def __init__(self, steps):
        if not hasattr(steps[-1][-1], 'predict_proba'):
            raise ValueError(
                "Variant pipelines must have a classification estimator"
                "with a 'predict_proba' method as their final step!")

        super().__init__(steps)

    def parse_preds(self, preds):
        if hasattr(self, 'classes_'):
            true_indx = [i for i, x in enumerate(self.classes_) if x]
            parse_preds = [scrs[true_indx] for scrs in preds]

        else:
            parse_preds = preds

        return parse_preds

    def predict_base(self, omic_data):
        return self.predict_proba(omic_data)

    def score(self, X, y=None):
        """Score the accuracy of the pipeline in predicting the given
           phenotype. Used eg. to ensure compatibility with cross-validation
           methods implemented in sklearn.

        Parameters
        ----------
        X : array-like, shape (n_samples,n_features)
            An expression dataset.

        y : array-like, shape (n_samples,)
            A boolean vector corresponding to the presence of a particular
            type of mutation in the same set of samples as the given
            expression dataset.

        Returns
        -------
        S : float
            The AUC score corresponding to mutation classification accuracy.

        """
        return roc_auc_score(y, self.predict_omic(X))


class ValuePipe(OmicPipe):
    """A class corresponding to pipelines which use continuous data to
       predict continuous outcomes.
    """
    pass


class VariantPipe(LabelPipe):
    """A class corresponding to pipelines for predicting discrete gene
       mutation states such as SNPs, indels, and frameshifts.
    """

    def __init__(self, steps, path_keys=None):
        super().__init__(steps)
        self.path_keys = path_keys

    def __repr__(self):
        """Prints the classifier name and the feature selection path key."""
        return '{}_{}'.format(
            type(self).__name__, str(self.get_params()['path_keys']))


class MutPipe(UniPipe, VariantPipe):
    """A class corresponding to pipelines for predicting
       discrete gene mutation states individually.
    """

    cvSplitMethod = MutShuffleSplit

    @classmethod
    def extra_fit_params(cls, cohort):
        return {'mut_genes': cohort.mut_genes,
                'path_obj': cohort.path}


class MultiVariantPipe(MultiPipe, VariantPipe):
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

    @classmethod
    def get_scores(cls, estimator, expr, mut_list):
        if len(mut_list[0]) == estimator.named_steps['fit'].task_count:
            mut_list = np.array(mut_list).transpose().tolist()

        pred_y = estimator.predict_mut(expr)
        auc_scores = [roc_auc_score(actl, pred)
                      for actl, pred in zip(mut_list, pred_y)]

        return auc_scores

    @classmethod
    def score_mut(cls, estimator, expr, mut_list):
        """Computes the AUC score using the classifier on a expr-mut pair.

        Parameters
        ----------
        expr : array-like, shape (n_samples,n_features)
            An expression dataset.

        estimator : 

        mut_list : array-like, shape (n_samples,)
            A boolean vector corresponding to the presence of a particular
            type of mutation in the same set of samples as the given
            expression dataset.

        Returns
        -------
        S : float
            The AUC score corresponding to mutation classification accuracy.
        """
        return np.min(cls.get_scores(estimator, expr, mut_list))

    def tune_coh(self,
                 cohort, pheno,
                 tune_splits=2, test_count=16,
                 include_samps=None, exclude_samps=None,
                 include_genes=None, exclude_genes=None,
                 verbose=False):
        """Tunes the pipeline by sampling over the tuning parameters."""

        samps, genes = cohort._validate_dims(exclude_samps=exclude_samps,
                                             gene_list=gene_list)
        expr = cohort.train_expr_.loc[samps, genes]
        mut_list = [cohort.train_mut_.status(samps, mtype)
                    for mtype in mtypes]

        # get internal cross-validation splits in the training set and use
        # them to tune the classifier
        tune_cvs = MutShuffleSplit(
            n_splits=tune_splits, test_size=0.2,
            random_state=(cohort.cv_seed ** 2) % 42949672)

        # checks if the classifier has parameters to be tuned
        if self.tune_priors:
            prior_counts = [len(x) if hasattr(x, '__len__') else float('Inf')
                            for x in self.cur_tuning.values()]
            max_tests = reduce(mul, prior_counts, 1)
            test_count = min(test_count, max_tests)

            # samples parameter combinations and tests each one
            grid_test = MutRandomizedCV(
                estimator=self, param_distributions=self.cur_tuning,
                fit_params={'mut_genes': cohort.mut_genes,
                            'path_obj': cohort.path,
                            'path_keys': path_keys},
                n_iter=test_count, scoring=self.score_mut,
                cv=tune_cvs, n_jobs=-1, refit=False
                )
            grid_test.fit(expr, mut_list)

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
                include_samples=None, exclude_samples=None,
                include_genes=None, exclude_genes=None):
        """Fits a classifier."""
        samps, genes = cohort._validate_dims(exclude_samps=exclude_samps,
                                             gene_list=gene_list)
        mut_list = [cohort.train_mut_.status(samps, mtype)
                    for mtype in mtypes]

        return self.fit(
            X=cohort.train_expr_.loc[samps, genes], y=mut_list,
            verbose=verbose,
            **{'mut_genes': cohort.mut_genes,
               'path_obj': cohort.path,
               'path_keys': path_keys}
            )

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

        score_cvs = MutShuffleSplit(
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

    def eval_coh(self,
                 cohort, pheno,
                 include_samples=None, exclude_samples=None,
                 include_genes=None, exclude_genes=None):
        """Evaluate the performance of a classifier."""
        samps, genes = cohort._validate_dims(
            exclude_samps=exclude_samps, gene_list=gene_list, use_test=True)

        return self.score_mut(
            self,
            cohort.test_expr_.loc[samps, genes],
            [cohort.test_mut_.status(samps, mtype) for mtype in mtypes]
            )

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


class DrugPipe(UniPipe, ValuePipe):
    """A class corresponding to pipelines for predicting drug sensitivity
       using expression data.
    """

    cvSplitMethod = DrugShuffleSplit
