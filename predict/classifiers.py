
"""
Hetman (Heterogeneity Manifold)
Prediction of mutation sub-types using expression data.
This file contains the algorithms used to predict discrete mutation states.
"""

# Author: Michal Grzadkowski <grzadkow@ohsu.edu>

from .pipelines import UniVariantPipe, MultiVariantPipe
from .selection import PathwaySelect

from math import exp
from scipy import stats

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier


# .. classifiers that don't use any prior information ..
class NaiveBayes(UniVariantPipe):
    """A class corresponding to Gaussian Naive Bayesian classification
       of mutation status.
    """

    def __init__(self, path_keys=None):
        feat_step = PathwaySelect(path_keys=path_keys)
        norm_step = StandardScaler()
        fit_step = GaussianNB()
        UniVariantPipe.__init__(
            self, steps=[
                ('feat', feat_step), ('norm', norm_step), ('fit', fit_step)],
            path_keys=path_keys
            )


class RobustNB(UniVariantPipe):
    """A class corresponding to Gaussian Naive Bayesian classification
       of mutation status with robust feature scaling.
    """

    def __init__(self, path_keys=None):
        feat_step = PathwaySelect(path_keys=path_keys)
        norm_step = RobustScaler()
        fit_step = GaussianNB()
        UniVariantPipe.__init__(
            self,
            [('feat', feat_step), ('norm', norm_step), ('fit', fit_step)],
            path_keys
            )


class Lasso(UniVariantPipe):
    """A class corresponding to logistic regression classification
       of mutation status with the lasso regularization penalty.
    """

    tune_priors = (
        ('fit__C', stats.lognorm(scale=exp(-1), s=exp(1))),
    )

    def __init__(self, path_keys=None):
        feat_step = PathwaySelect(path_keys=path_keys)
        norm_step = StandardScaler()
        fit_step = LogisticRegression(
            penalty='l1', tol=1e-2, class_weight='balanced')
        UniVariantPipe.__init__(self,
                                [('feat', feat_step), ('norm', norm_step), ('fit', fit_step)],
                                path_keys)


class LogReg(UniVariantPipe):
    """A class corresponding to logistic regression classification
       of mutation status with the elastic net regularization penalty.
    """

    tune_priors = (
        ('fit__alpha', stats.lognorm(scale=exp(1), s=exp(1))),
        ('fit__l1_ratio', (0.05, 0.25, 0.5, 0.75, 0.95))
    )

    def __init__(self, path_keys=None):
        feat_step = PathwaySelect(path_keys=path_keys)
        norm_step = StandardScaler()
        fit_step = SGDClassifier(
            loss='log', penalty='elasticnet',
            n_iter=100, class_weight='balanced')
        UniVariantPipe.__init__(self,
                                [('feat', feat_step), ('norm', norm_step), ('fit', fit_step)],
                                path_keys)


class Ridge(UniVariantPipe):
    """A class corresponding to logistic regression classification
       of mutation status with the ridge regularization penalty.
    """

    tune_priors = (
        ('fit__C', stats.lognorm(scale=exp(-1), s=exp(1))),
    )

    def __init__(self, path_keys=None):
        feat_step = PathwaySelect(path_keys=path_keys)
        norm_step = StandardScaler()
        fit_step = LogisticRegression(
            penalty='l1', tol=1e-2, class_weight='balanced')
        UniVariantPipe.__init__(self,
                                [('feat', feat_step), ('norm', norm_step), ('fit', fit_step)],
                                path_keys)


class SVCpoly(UniVariantPipe):
    """A class corresponding to C-support vector classification
       of mutation status with a radial basis kernel.
    """
   
    tune_priors = (
        ('fit__C', stats.lognorm(scale=exp(-1), s=exp(1))),
        ('fit__coef0', [-2., -1., -0.5, 0., 0.5, 1., 2.]),
    )

    def __init__(self, path_keys=None):
        feat_step = PathwaySelect(path_keys=path_keys)
        norm_step = StandardScaler()
        fit_step = SVC(
            kernel='poly', probability=True, degree=2,
            cache_size=500, class_weight='balanced')
        UniVariantPipe.__init__(self,
                                [('feat', feat_step), ('norm', norm_step), ('fit', fit_step)],
                                path_keys)


class SVCrbf(UniVariantPipe):
    """A class corresponding to C-support vector classification
       of mutation status with a radial basis kernel.
    """
   
    tune_priors = (
        ('fit__C', stats.lognorm(scale=exp(-1), s=exp(2))),
        ('fit__gamma', stats.lognorm(scale=1e-4, s=exp(2)))
    )

    def __init__(self, path_keys=None):
        feat_step = PathwaySelect(path_keys=path_keys)
        norm_step = StandardScaler()
        fit_step = SVC(
            kernel='rbf', probability=True,
            cache_size=500, class_weight='balanced')
        UniVariantPipe.__init__(self,
                                [('feat', feat_step), ('norm', norm_step), ('fit', fit_step)],
                                path_keys)


class rForest(UniVariantPipe):
    """A class corresponding to random forest classification
       of mutation status.
    """

    tune_priors = (
        ('fit__max_features', (0.005, 0.01, 0.02, 0.04, 0.08, 0.15)),
        ('fit__min_samples_leaf', (0.0001, 0.01, 0.05))
    )

    def __init__(self, path_keys=None):
        feat_step = PathwaySelect(path_keys=path_keys)
        norm_step = StandardScaler()
        fit_step = RandomForestClassifier(
                    n_estimators=500, class_weight='balanced')
        UniVariantPipe.__init__(self,
                                [('feat', feat_step), ('norm', norm_step), ('fit', fit_step)],
                                path_keys)


class KNeigh(UniVariantPipe):
    """A class corresponding to k-nearest neighbours voting classification
       of mutation status.
    """
    
    tune_priors = (
        ('fit__n_neighbors', (4, 8, 16, 25, 40)),
    )

    def __init__(self, path_keys=None):
        feat_step = PathwaySelect(path_keys=path_keys)
        norm_step = StandardScaler()
        fit_step = KNeighborsClassifier(
            weights='distance', algorithm='ball_tree')
        UniVariantPipe.__init__(self,
                                [('feat', feat_step), ('norm', norm_step), ('fit', fit_step)],
                                path_keys)


class GBCrbf(UniVariantPipe):
    """A class corresponding to gaussian process classification
       of mutation status with a radial basis kernel.
    """

    def __init__(self, mut_genes=None, expr_genes=None):
        self._tune_priors = {}
        norm_step = StandardScaler()
        fit_step = GaussianProcessClassifier()
        UniVariantPipe.__init__(self,
                                [('norm', norm_step), ('fit', fit_step)])

