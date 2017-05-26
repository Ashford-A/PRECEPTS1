
"""
HetMan (Heterogeneity Manifold)
Prediction of mutation sub-types using expression data.
This file contains classes used to organize feature selection, normalization,
and prediction methods into robust pipelines.
"""

# Author: Michal Grzadkowski <grzadkow@ohsu.edu>

from .cross_validation import MutShuffleSplit, cross_val_predict_mut

from abc import abstractmethod
import numpy as np

from numbers import Number
from functools import reduce
from operator import mul

from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score


class CohortPipe(Pipeline):
    """A class corresponding to pipelines for predicting genotypic and
       phenotypic features from expresssion data.
    """

    # the parameters that are to be tuned, with either statistical
    # distributions or iterables to be sampled from as values
    tune_priors = {}

    def __init__(self, steps, path_keys):
        super(CohortPipe, self).__init__(steps)
        self.set_params(path_keys=path_keys)
        self.cur_tuning = dict(self.tune_priors)

    def __repr__(self):
        """Prints the classifier name and path key."""
        return '{}_{}'.format(
            type(self).__name__, str(self.get_params()['path_keys']))

    def __str__(self):
        """Prints the tuned parameters of the pipeline."""
        param_str = type(self).__name__ + ' with '

        if self.tune_priors:
            param_list = self.get_params()
            param_str += reduce(
                lambda x, y: x + ', ' + y,
                [k + ': ' + '%s' % float('%.4g' % param_list[k])
                 if isinstance(param_list[k], Number)
                 else k + ': ' + param_list[k]
                 for k in self.cur_tuning.keys()]
                )
        else:
            param_str += 'no tuned parameters.'

        return param_str

    @abstractmethod
    def score_mut(cls, estimator, expr, mut):
        """Score the accuracy of the pipeline in predicting the state
           of a given set of mutations. Used to ensure compatibility with
           scoring methods implemented in sklearn.
        """

    # TODO: make this an abstract method, extract coefs from all classifiers
    def get_coef(self):
        """Gets the coefficients of the classifier."""
        return dict(zip(self.named_steps['feat'].expr_genes,
                        self.named_steps['fit'].coef_[0]))


class VariantPipe(CohortPipe):
    """A class corresponding to pipelines for predicting discrete gene
       mutation states such as SNPs, indels, and frameshifts.
    """

    def __init__(self, steps, path_keys):
        if not hasattr(steps[-1][-1], 'predict_proba'):
            raise ValueError(
                "Variant pipelines must have a classification estimator"
                "with a 'predict_proba' method as their final step!")
        super(VariantPipe, self).__init__(steps, path_keys)

    def predict_mut(self, expr):
        """Returns the probability of mutation presence calculated by the
           classifier based on the given expression matrix.
        """
        mut_scores = self.predict_proba(expr)
        if hasattr(self, 'classes_'):
            true_indx = [i for i, x in enumerate(self.classes_) if x]
        else:
            wghts = tuple(self.named_steps['fit'].weights_)
            true_indx = wghts.index(min(wghts))
        return [m[true_indx] for m in mut_scores]

    def predict_coh(self,
                    cohort, use_test=False, gene_list=None,
                    exclude_samps=None):
        """Predicts mutation status using a classifier."""
        samps, genes = cohort._validate_dims(exclude_samps=exclude_samps,
                                             gene_list=gene_list,
                                             use_test=use_test)

        if use_test:
            muts = self.predict_mut(expr=cohort.test_expr_.loc[samps, genes])
        else:
            muts = self.predict_mut(expr=cohort.train_expr_.loc[samps, genes])

        return muts

    def eval_coh(self,
                 cohort, mtype, gene_list=None, exclude_samps=None):
        """Evaluate the performance of a classifier."""
        samps, genes = cohort._validate_dims(
            exclude_samps=exclude_samps, gene_list=gene_list, use_test=True)

        return self.score_mut(self,
                              cohort.test_expr_.loc[samps, genes],
                              cohort.test_mut_.status(samps, mtype))


class UniVariantPipe(VariantPipe):
    """A class corresponding to pipelines for predicting
       discrete gene mutation states individually.
    """

    @classmethod
    def score_mut(cls, estimator, expr, mut):
        """Computes the AUC score using the classifier on a expr-mut pair.

        Parameters
        ----------
        expr : array-like, shape (n_samples,n_features)
            An expression dataset.
            
        estimator : 

        mut : array-like, shape (n_samples,)
            A boolean vector corresponding to the presence of a particular
            type of mutation in the same set of samples as the given
            expression dataset.

        Returns
        -------
        S : float
            The AUC score corresponding to mutation classification accuracy.
        """
        mut_scores = estimator.predict_mut(expr)
        return roc_auc_score(mut, mut_scores)

    def tune_coh(self,
                 cohort, mtype, gene_list=None, exclude_samps=None,
                 tune_splits=2, test_count=16, verbose=False):
        """Tunes the pipeline by sampling over the tuning parameters."""

        samps, genes = cohort._validate_dims(exclude_samps=exclude_samps,
                                             gene_list=gene_list)
        expr = cohort.train_expr_.loc[samps, genes]
        muts = cohort.train_mut_.status(samps, mtype)

        # get internal cross-validation splits in the training set and use
        # them to tune the classifier
        tune_cvs = MutShuffleSplit(
            n_splits=tune_splits, test_size=0.2,
            random_state=(cohort.intern_cv_ ** 2) % 42949672)

        # checks if the classifier has parameters to be tuned
        if self.tune_priors:
            prior_counts = [len(x) if hasattr(x, '__len__') else float('Inf')
                            for x in self.cur_tuning.values()]
            max_tests = reduce(mul, prior_counts, 1)
            test_count = min(test_count, max_tests)

            # samples parameter combinations and tests each one
            grid_test = RandomizedSearchCV(
                estimator=self, param_distributions=self.cur_tuning,
                fit_params={'feat__mut_genes': cohort.mut_genes,
                            'feat__path_obj': cohort.path_},
                n_iter=test_count, scoring=self.score_mut,
                cv=tune_cvs, n_jobs=-1, refit=False
                )
            grid_test.fit(expr, muts)

            # finds the best parameter combination and updates the classifier
            tune_scores = (grid_test.cv_results_['mean_test_score']
                           - grid_test.cv_results_['std_test_score'])
            self.set_params(
                **grid_test.cv_results_['params'][tune_scores.argmax()])
            if verbose:
                print(self)

        return self

    def fit_coh(self, cohort, mtype, gene_list=None, exclude_samps=None):
        """Fits a classifier."""
        samps, genes = cohort._validate_dims(exclude_samps=exclude_samps,
                                             gene_list=gene_list)
        muts = cohort.train_mut_.status(samps, mtype)

        return self.fit(X=cohort.train_expr_.loc[samps, genes],
                        y=cohort.train_mut_.status(samps, mtype),
                        feat__mut_genes=list(
                            reduce(lambda x,y: x|y, mtype.child.keys())),
                        feat__path_obj=cohort.path_)

    def score_coh(self,
                  cohort, mtype, score_splits=16,
                  gene_list=None, exclude_samps=None):
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
            fit_params={'feat__mut_genes': list(
                reduce(lambda x,y: x|y, mtype.child.keys())),
                        'feat__path_obj': cohort.path_},
            scoring=self.score_mut, cv=score_cvs, n_jobs=-1
            ), 25)

    def infer_coh(self,
                  cohort, mtype, infer_splits=16,
                  gene_list=None, exclude_samps=None):
        samps, genes = cohort._validate_dims(gene_list=gene_list)

        infer_scores = cross_val_predict_mut(
            estimator=self,
            X=cohort.train_expr_.loc[:, genes],
            y=cohort.train_mut_.status(samps, mtype),
            exclude_samps=exclude_samps, cv_fold=4, cv_count=infer_splits,
            fit_params={'feat__mut_genes': list(
                reduce(lambda x,y: x|y, mtype.child.keys())),
                        'feat__path_obj': cohort.path_},
            random_state=int(cohort.intern_cv_ ** 1.5) % 42949672, n_jobs=-1
            )

        return infer_scores


class MultiVariantPipe(VariantPipe):
    """A class corresponding to pipelines for predicting a collection of
       discrete gene mutation states simultaenously.
    """

    @classmethod
    def score_mut(cls, estimator, expr, mut_list):
        """Computes the AUC score using the classifier on a expr-mut pair.

        Parameters
        ----------
        expr : array-like, shape (n_samples,n_features)
            An expression dataset.

        estimator : 

        mut : array-like, shape (n_samples,)
            A boolean vector corresponding to the presence of a particular
            type of mutation in the same set of samples as the given
            expression dataset.

        Returns
        -------
        S : float
            The AUC score corresponding to mutation classification accuracy.
        """
        mut_scores = estimator.predict_mut(expr)
        return min([roc_auc_score(actl, pred)
                    for actl, pred in zip(mut_list, mut_scores)])

    def tune_coh(self,
                 cohort, mtypes, gene_list=None, exclude_samps=None,
                 tune_splits=2, test_count=16, verbose=False):
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
            random_state=(cohort.intern_cv_ ** 2) % 42949672)

        # checks if the classifier has parameters to be tuned
        if self.tune_priors:
            prior_counts = [len(x) if hasattr(x, '__len__') else float('Inf')
                            for x in self.cur_tuning.values()]
            max_tests = reduce(mul, prior_counts, 1)
            test_count = min(test_count, max_tests)

            # samples parameter combinations and tests each one
            grid_test = RandomizedSearchCV(
                estimator=self, param_distributions=self.cur_tuning,
                fit_params={'feat__mut_genes': cohort.mut_genes,
                            'feat__path_obj': cohort.path_},
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

    def fit_coh(self, cohort, mtypes, gene_list=None, exclude_samps=None):
        """Fits a classifier."""
        samps, genes = cohort._validate_dims(exclude_samps=exclude_samps,
                                             gene_list=gene_list)
        mut_list = [cohort.train_mut_.status(samps, mtype)
                    for mtype in mtypes]

        return self.fit(
            X=cohort.train_expr_.loc[samps, genes], y=mut_list,
            feat__mut_genes=reduce(lambda x, y: x | y,
                                   [set(dict(mtype)) for mtype in mtypes]),
            feat__path_obj=cohort.path_
            )

    def score_coh(self,
                  cohort, mtype, score_splits=16,
                  gene_list=None, exclude_samps=None):
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
            fit_params={'feat__mut_genes': list(
                reduce(lambda x, y: x | y, mtype.child.keys())),
                'feat__path_obj': cohort.path_},
            scoring=self.score_mut, cv=score_cvs, n_jobs=-1
            ), 25)

    def infer_coh(self,
                  cohort, mtype, infer_splits=16,
                  gene_list=None, exclude_samps=None):
        samps, genes = cohort._validate_dims(gene_list=gene_list)

        infer_scores = cross_val_predict_mut(
            estimator=self,
            X=cohort.train_expr_.loc[:, genes],
            y=cohort.train_mut_.status(samps, mtype),
            exclude_samps=exclude_samps, cv_fold=4, cv_count=infer_splits,
            fit_params={'feat__mut_genes': list(
                reduce(lambda x, y: x | y, mtype.child.keys())),
                'feat__path_obj': cohort.path_},
            random_state=int(cohort.intern_cv_ ** 1.5) % 42949672, n_jobs=-1
            )

        return infer_scores

