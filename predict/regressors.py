
from math import exp
from scipy import stats

from .pipelines import DrugPipe

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet as ENet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor


class ElasticNet(DrugPipe):
    """A class corresponding to elastic net regression
       of gene gain/loss status.
    """

    tune_priors = (
        ('fit__alpha', stats.lognorm(scale=exp(1), s=exp(1))),
        ('fit__l1_ratio', (0.05, 0.25, 0.5, 0.75, 0.95))
        )

    # TODO: consider how to do feature selection for drug pipelines
    # i.e. for variantpipe there is a pathway feature selection method
    # based on the neighborhoods of the genes with alterations
    def __init__(self):
        norm_step = StandardScaler()
        fit_step = ENet(normalize=False, max_iter=5000)
        DrugPipe.__init__(self, [('norm', norm_step), ('fit', fit_step)])


class SVRrbf(DrugPipe):
    """A class corresponding to Support Vector Machine regression
       of gene gain/loss status using a radial basis kernel.
    """

    tune_priors = (
        ('fit__C', stats.lognorm(scale=exp(-1), s=exp(1))),
        ('fit__gamma', stats.lognorm(scale=1e-5, s=exp(2)))
        )

    def __init__(self):
        norm_step = StandardScaler()
        fit_step = SVR(kernel='rbf', cache_size=500)
        DrugPipe.__init__(self, [('norm', norm_step), ('fit', fit_step)])


class rForest(DrugPipe):
    """A class corresponding to Random Forest regression
       of gene gain/loss status.
    """

    tune_priors = (
        ('fit__min_samples_leaf', (0.001, 0.005, 0.01, 0.05)),
        ('fit__max_features', (5, 'sqrt', 'log2', 0.02))
        )

    def __init__(self):
        norm_step = StandardScaler()
        fit_step = RandomForestRegressor(n_estimators=1000)
        DrugPipe.__init__(self, [('norm', norm_step), ('fit', fit_step)])
