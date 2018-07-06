
from ....predict.pipelines import PresencePipe
from ....predict.stan.base import *
from ....predict.stan.logistic.classifiers import BaseLogistic
from ....predict.stan.logistic.stan_models import gauss_model, cauchy_model

import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler


class OptimModel(BaseLogistic, StanOptimizing):

    def run_model(self, **fit_params):
        super().run_model(**{**fit_params, **{'iter': 1e4}})


class Base(PresencePipe):

    tune_priors = (
        ('fit__alpha', tuple(10 ** np.linspace(-4, 1.75, 24))),
        )
 
    norm_inst = StandardScaler()
    fit_inst = OptimModel(model_code=gauss_model)

    def __init__(self):
        super().__init__([('norm', self.norm_inst), ('fit', self.fit_inst)])


class Cauchy(Base):

    fit_inst = OptimModel(model_code=cauchy_model)

