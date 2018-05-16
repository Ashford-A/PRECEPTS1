
from .....predict.pipelines import PresencePipe
from .....predict.stan.base import *
from .....predict.stan.margins.sym import SymMargins
from .....predict.stan.margins.stan_models import gauss_model as use_model

from scipy.stats import lognorm
from sklearn.preprocessing import RobustScaler


class UseOverlap(SymMargins):

    def __init__(self, model_code, alpha=0.01):
        super().__init__(model_code, lbl_var=1.0, alpha=alpha)

    def predict_proba(self, X):
        return self.calc_pred_labels(X)


class UseOptimizing(UseOverlap, StanOptimizing):
 
    def run_model(self, **fit_params):
        super().run_model(**{**fit_params, **{'iter': 1e4}})


class UseVariational(UseOverlap, StanVariational):
    pass


class UseSampling(UseOverlap, StanSampling):

    def run_model(self, **fit_params):
        super().run_model(**{**fit_params, **{'iter': 100}})


class UsePipe(PresencePipe):

    tune_priors = (
        ('fit__alpha', lognorm(scale=1e-2, s=2)),
        )

    def __init__(self, fit_inst):
        self.fit_inst = fit_inst
        super().__init__([('norm', RobustScaler()), ('fit', self.fit_inst)])

