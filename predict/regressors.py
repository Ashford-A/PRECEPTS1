
"""Specific pipelines for prediction of continuous phenotypes.

See Also:
    :module:`.pipelines`: Base classes for phenotype prediction.
    :module:`.classifiers`: Predicting binary phenotypes.

Author: Michal Grzadkowski <grzadkow@ohsu.edu>

"""

from math import exp
from scipy import stats

from .pipelines import UniPipe, LinearPipe, ValuePipe
from .selection import *

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor as ENet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor



class SimpleRegressor(LinearPipe, ValuePipe):
    """A class for linear regression using ordinary least squares."""

    def __init__(self, path_keys=None):
        feat_step = PathwaySelect(path_keys=path_keys)
        norm_step = StandardScaler()
        fit_step = LinearRegression()

        super().__init__(
            [('feat', feat_step), ('norm', norm_step), ('fit', fit_step)],
            path_keys=path_keys
            )


class ElasticNet(LinearPipe, ValuePipe):
    """A class for linear regression using the Elastic Net penalty."""

    tune_priors = (
        #('fit__alpha', stats.lognorm(scale=exp(-2), s=exp(1))),
        ('fit__l1_ratio', (0.05, 0.25, 0.5, 0.75, 0.95)),
        )

    def __init__(self, path_keys=None):
        feat_step = PathwaySelect(path_keys=path_keys)
        norm_step = StandardScaler()
        fit_step = ENet(penalty='elasticnet', max_iter=5000, alpha=1e-4)

        super().__init__(
            [('feat', feat_step), ('norm', norm_step), ('fit', fit_step)],
            path_keys=path_keys
            )


class SVRrbf(UniPipe, ValuePipe):
    """A class for Support Vector Regression using a radial basis kernel."""

    tune_priors = (
        ('fit__C', stats.lognorm(scale=exp(-1), s=exp(1))),
        ('fit__gamma', stats.lognorm(scale=1e-5, s=exp(2)))
        )

    def __init__(self, path_keys=None):
        feat_step = PathwaySelect(path_keys=path_keys)
        norm_step = StandardScaler()
        fit_step = SVR(kernel='rbf', cache_size=500)

        super().__init__(
            [('feat', feat_step), ('norm', norm_step), ('fit', fit_step)],
            path_keys=path_keys
            )


class rForest(UniPipe, ValuePipe):
    """A class for regression using an ensemble of random decision trees."""

    tune_priors = (
        ('fit__min_samples_leaf', (0.001, 0.005, 0.01, 0.05)),
        ('fit__max_features', (5, 'sqrt', 'log2', 0.005, 0.01, 0.02))
        )

    def __init__(self, path_keys=None):
        feat_step = PathwaySelect(path_keys=path_keys)
        norm_step = StandardScaler()
        fit_step = RandomForestRegressor(n_estimators=4000)

        super().__init__(
            [('feat', feat_step), ('norm', norm_step), ('fit', fit_step)],
            path_keys=path_keys
            )


class GradientBoosting(UniPipe, ValuePipe):
    """A class for regression by building an additive ensemble of trees."""

    tune_priors = (
        ('fit__max_depth', (2, 3, 4, 5, 8)),
        ('fit__min_samples_split', (2, 4, 6, 10)),
        )

    def __init__(self, path_keys=None):
        feat_step = PathwaySelect(path_keys=path_keys)
        norm_step = StandardScaler()
        fit_step = GradientBoostingRegressor(n_estimators=200, loss='huber')

        super().__init__(
            [('feat', feat_step), ('norm', norm_step), ('fit', fit_step)],
            path_keys=path_keys
            )


class KNeighbors(UniPipe, ValuePipe):

    tune_priors = (
        ('fit__n_neighbors', (2, 4, 8, 12)),
        ('fit__p', (1, 2, 3, 5, 10))
        )

    def __init__(self, path_keys=None):
        feat_step = PathwaySelect(path_keys=path_keys)
        norm_step = StandardScaler()
        fit_step = KNeighborsRegressor(weights='distance')

        super().__init__(
            [('feat', feat_step), ('norm', norm_step), ('fit', fit_step)],
            path_keys=path_keys
            )


class GaussianProcess(UniPipe, ValuePipe):

    tune_priors = ()

    def __init__(self, path_keys=None):
        feat_step = PathwaySelect(path_keys=path_keys)
        norm_step = StandardScaler()
        fit_step = GaussianProcessRegressor(normalize_y=True)

        super().__init__(
            [('feat', feat_step), ('norm', norm_step), ('fit', fit_step)],
            path_keys=path_keys
            )

