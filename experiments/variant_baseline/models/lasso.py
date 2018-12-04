
from dryadic.learning.pipelines import PresencePipe
from dryadic.learning.selection import SelectMeanVar

import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score


class Base(PresencePipe):

    tune_priors = (
        ('fit__C', tuple(10 ** np.linspace(-3, 5.75, 36))),
        )

    feat_inst = SelectMeanVar(mean_perc=95, var_perc=95)
    norm_inst = StandardScaler()
    fit_inst = LogisticRegression(penalty='l1', max_iter=200,
                                  class_weight='balanced')

    def __init__(self):
        super().__init__([('feat', self.feat_inst), ('norm', self.norm_inst),
                          ('fit', self.fit_inst)])


class Meanvar(Base):

    tune_priors = (
        ('feat__mean_perc', (50, 80, 90, 95, 99, 100)),
        ('feat__var_perc', (50, 80, 90, 95, 99, 100)),
        )

    feat_inst = SelectMeanVar()
    fit_inst = LogisticRegression(penalty='l1', max_iter=200, C=0.1,
                                  class_weight='balanced')


class Norm_robust(Base):

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

