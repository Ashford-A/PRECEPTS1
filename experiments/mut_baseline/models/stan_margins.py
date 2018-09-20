
from dryadic.learning.pipelines import PresencePipe
from dryadic.learning.selection import SelectMeanVar

from dryadic.learning.stan.base import StanOptimizing
from dryadic.learning.stan.margins.classifiers import (
    GaussLabels, CauchyLabels)
from dryadic.learning.stan.margins.stan_models import *

import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler


class OptimModel(GaussLabels, StanOptimizing):

    def run_model(self, **fit_params):
        super().run_model(**{**fit_params, **{'iter': 5e4}})


class OptimCauchy(CauchyLabels, StanOptimizing):

    def run_model(self, **fit_params):
        super().run_model(**{**fit_params, **{'iter': 5e4}})


class Base(PresencePipe):

    tune_priors = (
        ('fit__alpha', tuple(10 ** np.linspace(-2, -3.75, 36))),
        )

    feat_inst = SelectMeanVar(mean_perc=95, var_perc=95)
    norm_inst = StandardScaler()
    fit_inst = OptimModel(model_code=gauss_model)

    def __init__(self):
        super().__init__([('feat', self.feat_inst), ('norm', self.norm_inst),
                          ('fit', self.fit_inst)])


class Cauchy(Base):

    fit_inst = OptimCauchy(model_code=cauchy_model)

