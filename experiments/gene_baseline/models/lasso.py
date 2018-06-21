
from ....predict.pipelines import PresencePipe
import numpy as np
from scipy import stats

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score


class Base(PresencePipe):

    tune_priors = (
        ('fit__C', tuple(10 ** np.linspace(-5.75, 5.75, 24))),
        )

    norm_inst = StandardScaler()
    fit_inst = LogisticRegression(penalty='l1', max_iter=500,
                                  class_weight='balanced')

    def __init__(self):
        super().__init__([('norm', self.norm_inst), ('fit', self.fit_inst)])


class Norm_robust(Base):

    tune_priors = (
        ('fit__C', tuple(10 ** np.linspace(-3.75, 7.75, 24))),
        )
    
    norm_inst = RobustScaler()


class Iter_short(Base):

    fit_inst = LogisticRegression(penalty='l1', max_iter=80,
                                  class_weight='balanced')


class Tune_aupr(Base):

    @staticmethod
    def score_pheno(actual_pheno, pred_pheno):
        pheno_score = 0.5

        if (len(np.unique(actual_pheno)) > 1
                and len(np.unique(pred_pheno)) > 1):
            pheno_score = average_precision_score(actual_pheno, pred_pheno)

        return pheno_score


class Tune_distr(Base):

    tune_priors = (
        ('fit__C', stats.lognorm(scale=0.1, s=4)),
        )

