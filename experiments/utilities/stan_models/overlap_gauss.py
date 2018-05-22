
from ....predict.pipelines import PresencePipe
from ....predict.stan.base import StanOptimizing
from ....predict.stan.margins.classifiers import GaussLabels
from ....predict.stan.margins.stan_models import gauss_model as use_model

from scipy.stats import lognorm
from sklearn.preprocessing import RobustScaler


class UseOverlap(GaussLabels):

    def __init__(self, model_code, alpha=0.01):
        super().__init__(model_code,
                         wt_distr=(-1.0, 1.0), mut_distr=(1.0, 1.0),
                         alpha=alpha)

    def predict_proba(self, X):
        return self.calc_pred_labels(X)


class UseModel(UseOverlap, StanOptimizing):
 
    def run_model(self, **fit_params):
        super().run_model(**{**fit_params, **{'iter': 2e4}})


class UsePipe(PresencePipe):

    tune_priors = (
        ('fit__alpha', lognorm(scale=1e-2, s=2)),
        )

    def __init__(self):
        self.fit_inst = UseModel(model_code=use_model)
        super().__init__([('norm', RobustScaler()), ('fit', self.fit_inst)])

