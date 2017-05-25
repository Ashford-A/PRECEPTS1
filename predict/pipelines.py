
"""
HetMan (Heterogeneity Manifold)
Prediction of mutation sub-types using expression data.
This file contains classes used to organize feature selection, normalization,
and prediction methods into robust pipelines.
"""

# Author: Michal Grzadkowski <grzadkow@ohsu.edu>

from abc import abstractmethod

from numbers import Number
from functools import reduce
from operator import mul

from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score


class MutPipe(Pipeline):
    """A class corresponding to pipelines for predicting gene mutations
       using expression data.
    """

    # the parameters that are to be tuned, with either statistical
    # distributions or iterables to be sampled from as values
    tune_priors = {}

    def __init__(self, steps, path_keys):
        super(MutPipe, self).__init__(steps)
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

    def tune(self, expr, mut, mut_genes, path_obj, cv_samples,
             test_count=16, verbose=False):
        """Tunes the pipeline by sampling over the tuning parameters."""

        # checks if the classifier has parameters to be tuned
        if self.tune_priors:
            prior_counts = [len(x) if hasattr(x, '__len__') else float('Inf')
                            for x in self.cur_tuning.values()]
            max_tests = reduce(mul, prior_counts, 1)
            test_count = min(test_count, max_tests)

            # samples parameter combinations and tests each one
            grid_test = RandomizedSearchCV(
                estimator=self, param_distributions=self.cur_tuning,
                fit_params={'feat__mut_genes': mut_genes,
                            'feat__path_obj': path_obj},
                n_iter=test_count, scoring=self.score_mut,
                cv=cv_samples, n_jobs=-1, refit=False
                )
            grid_test.fit(expr, mut)

            # finds the best parameter combination and updates the classifier
            tune_scores = (grid_test.cv_results_['mean_test_score']
                           - grid_test.cv_results_['std_test_score'])
            self.set_params(
                **grid_test.cv_results_['params'][tune_scores.argmax()])
            if verbose:
                print(self)

        return self

    def get_coef(self):
        """Gets the coefficients of the classifier."""
        return dict(zip(self.named_steps['feat'].expr_genes,
                        self.named_steps['fit'].coef_[0]))


class ClassPipe(MutPipe):
    """A class corresponding to pipelines for predicting
       discrete gene mutation states.
    """

    def __init__(self, steps, path_keys):
        if not hasattr(steps[-1][-1], 'predict_proba'):
            raise ValueError(
                "Classifier pipelines must have a classification estimator"
                "with a 'predict_proba' method as their final step!")
        super(ClassPipe, self).__init__(steps, path_keys)

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

