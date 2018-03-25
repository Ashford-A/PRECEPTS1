
"""Simple classifiers from predicting -omic phenotypes.

Author: Michal Grzadkowski <grzadkow@ohsu.edu>

"""

from ..pipelines import PresencePipe, LinearPipe, EnsemblePipe

from math import exp
from scipy import stats

from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier


class BasicPipe(PresencePipe):

    def __init__(self, fit_inst):
        super().__init__([('norm', StandardScaler()), ('fit', fit_inst)])


class NaiveBayes(BasicPipe):
    """A class corresponding to Gaussian Naive Bayesian classification
       of mutation status.
    """

    def __init__(self):
        super().__init__(GaussianNB())


class Lasso(BasicPipe, LinearPipe):
    """A class corresponding to logistic regression classification
       of mutation status with the lasso regularization penalty.
    """

    tune_priors = (
        ('fit__C', stats.lognorm(scale=exp(-1), s=exp(1))),
        )

    def __init__(self):
        super().__init__(LogisticRegression(penalty='l1', max_iter=500,
                                            class_weight='balanced'))


class Ridge(BasicPipe, LinearPipe):
    """A class corresponding to logistic regression classification
       of mutation status with the ridge regularization penalty.
    """

    tune_priors = (
        ('fit__C', stats.lognorm(scale=exp(-1), s=exp(1))),
        )

    def __init__(self):
        super().__init__(LogisticRegression(penalty='l2', max_iter=1000,
                                            class_weight='balanced'))


class ElasticNet(BasicPipe, LinearPipe):
    """A class corresponding to logistic regression classification
       of mutation status with the elastic net regularization penalty.
    """

    tune_priors = (
        ('fit__alpha', stats.lognorm(scale=exp(1), s=exp(1))),
        ('fit__l1_ratio', (0.05, 0.25, 0.5, 0.75, 0.95)),
        )

    def __init__(self):
        super().__init__(
            SGDClassifier(loss='log', penalty='elasticnet',
                          max_iter=1000, class_weight='balanced')
            )


class SVCpoly(BasicPipe):
    """A class corresponding to C-support vector classification
       of mutation status with a radial basis kernel.
    """
   
    tune_priors = (
        ('fit__C', stats.lognorm(scale=exp(-1), s=exp(1))),
        ('fit__coef0', (-5., -2., -1., -0.5, 0., 0.5, 1., 2., 5.)),
        ('fit__degree', (2, 3, 4, 5)),
        )

    def __init__(self):
        super().__init__(SVC(kernel='poly', probability=True,
                             cache_size=500, class_weight='balanced'))


class SVCrbf(BasicPipe):
    """A class corresponding to C-support vector classification
       of mutation status with a radial basis kernel.
    """
   
    tune_priors = (
        ('fit__C', stats.lognorm(scale=exp(-1), s=exp(2))),
        ('fit__gamma', stats.lognorm(scale=1e-4, s=exp(2))),
        )

    def __init__(self):
        super().__init__(SVC(kernel='rbf', probability=True,
                             cache_size=500, class_weight='balanced'))


class GradBoost(BasicPipe, EnsemblePipe):
    """A class for classification using an additive ensemble of trees."""
    
    tune_priors = (
        ('fit__max_depth', (2, 3, 4, 5, 8)),
        ('fit__min_samples_split', (0.005, 0.01, 0.02, 0.03, 0.04, 0.05)),
        )

    def __init__(self):
        super().__init__(GradientBoostingClassifier(n_estimators=200))


class RandomForest(BasicPipe, EnsemblePipe):
    """A class corresponding to random forest classification
       of mutation status.
    """

    tune_priors = (
        ('fit__max_features', (0.01, 0.02, 0.04, 0.08, 0.15, "sqrt", "log2")),
        ('fit__min_samples_leaf', (0.0001, 0.005, 0.01, 0.02, 0.03, 0.05)),
        )

    def __init__(self):
        super().__init__(RandomForestClassifier(n_estimators=500,
                                                class_weight='balanced'))


class KNeigh(BasicPipe):
    """A class corresponding to k-nearest neighbours voting classification
       of mutation status.
    """
    
    tune_priors = (
        ('fit__n_neighbors', (4, 8, 16, 25, 40)),
        )

    def __init__(self):
        super().__init__(KNeighborsClassifier(weights='distance',
                                              algorithm='ball_tree'))


class GBCrbf(BasicPipe):
    """A class corresponding to gaussian process classification
       of mutation status with a radial basis kernel.
    """

    def __init__(self):
        super().__init__(GaussianProcessClassifier())

