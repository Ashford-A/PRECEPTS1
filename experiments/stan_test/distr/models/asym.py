
from .....predict.pipelines import PresencePipe
from .....predict.stan.base import *
from .....predict.stan.margins.asym import AsymMargins
from .....predict.stan.margins.stan_models import cauchy_model as use_model

from scipy.stats import lognorm
from sklearn.preprocessing import StandardScaler


class UseOptimizing(AsymMargins, StanOptimizing):
    pass


class UseVariational(AsymMargins, StanVariational):
    pass


class UseSampling(AsymMargins, StanSampling):
    pass


class UsePipe(PresencePipe):
    
    tune_priors = (
        ('fit__alpha', lognorm(scale=1e-2, s=3)),
        ('fit__wt_distr', ((-1, 0.1), (-1, 0.5))),
        ('fit__mut_distr', ((1, 0.1), (1, 0.5))),
        )

    def __init__(self, fit_inst):
        self.fit_inst = fit_inst
        super().__init__([('norm', StandardScaler()), ('fit', self.fit_inst)])

