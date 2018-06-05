
from .....predict.pipelines import PresencePipe
from .....predict.stan.base import *
from .....predict.stan.logistic.classifiers import BaseLogistic
from .....predict.stan.logistic.stan_models import gauss_model as use_model

from scipy.stats import lognorm
from sklearn.preprocessing import RobustScaler


class UseLogistic(BaseLogistic):

    def predict_proba(self, X):
        return self.calc_pred_labels(X)


class UseOptimizing(UseLogistic, StanOptimizing):
 
    def run_model(self, **fit_params):
        super().run_model(**{**fit_params, **{'iter': 1e4}})


class UseVariational(UseLogistic, StanVariational):
    pass


class UseSampling(UseLogistic, StanSampling):

    def run_model(self, **fit_params):
        super().run_model(**{**fit_params, **{'iter': 250}})


class UsePipe(PresencePipe):

    tune_priors = (
        ('fit__alpha', lognorm(scale=1e-2, s=2)),
        )

    def __init__(self, fit_inst):
        self.fit_inst = fit_inst
        super().__init__([('norm', RobustScaler()), ('fit', self.fit_inst)])

