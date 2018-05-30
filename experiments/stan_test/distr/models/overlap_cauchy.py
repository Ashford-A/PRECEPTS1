
from .....predict.pipelines import PresencePipe
from .....predict.stan.base import *
from .....predict.stan.margins.classifiers import CauchyLabels
from .....predict.stan.margins.stan_models import cauchy_model as use_model

from scipy.stats import lognorm
from sklearn.preprocessing import RobustScaler


class UseOverlap(CauchyLabels):

    def __init__(self, model_code, alpha=0.01):
        super().__init__(model_code,
                         wt_distr=(-1.0, 1.0), mut_distr=(1.0, 1.0),
                         alpha=alpha)

    def predict_proba(self, X):
        return self.calc_pred_labels(X)


class UseOptimizing(UseOverlap, StanOptimizing):
 
    def run_model(self, **fit_params):
        super().run_model(**{**fit_params, **{'iter': 1e5}})


class UseVariational(UseOverlap, StanVariational):
    pass


class UseSampling(UseOverlap, StanSampling):

    def run_model(self, **fit_params):
        super().run_model(**{**fit_params, **{'iter': 150}})


class UsePipe(PresencePipe):

    tune_priors = (
        ('fit__alpha', lognorm(scale=1e-2, s=2)),
        )

    def __init__(self, fit_inst):
        self.fit_inst = fit_inst
        super().__init__([('norm', RobustScaler()), ('fit', self.fit_inst)])

