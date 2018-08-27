
from dryadic.learning.pipelines import PresencePipe
from dryadic.learning.selection import SelectMeanVar

import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import SGDClassifier


class Base(PresencePipe):

    tune_priors = (
        ('fit__alpha', tuple(10 ** np.linspace(-3.5, -0.2, 12))),
        ('fit__l1_ratio', (0.25, 0.5, 0.75)),
        )

    feat_inst = SelectMeanVar(mean_perc=95, var_perc=95)
    norm_inst = StandardScaler()
    fit_inst = SGDClassifier(loss='log', penalty='elasticnet',
                             max_iter=1000, class_weight='balanced')

    def __init__(self):
        super().__init__([('feat', self.feat_inst), ('norm', self.norm_inst),
                          ('fit', self.fit_inst)])


class Norm_robust(Base):

    tune_priors = (
        ('fit__alpha', tuple(10 ** np.linspace(-5, 0.5, 12))),
        ('fit__l1_ratio', (0.25, 0.5, 0.75)),
        )

    norm_inst = RobustScaler()


class Ratio_skew(Base):

    tune_priors = (
        ('fit__alpha', tuple(10 ** np.linspace(-3.5, -0.3, 9))),
        ('fit__l1_ratio', (0.15, 0.3, 0.45, 0.6)),
        )


class Ratio_fixed(Base):

    tune_priors = (
        ('fit__alpha', tuple(10 ** np.linspace(-3.2, -0.4, 36))),
        )

    fit_inst = SGDClassifier(l1_ratio=0.4, loss='log', penalty='elasticnet',
                             max_iter=1000, class_weight='balanced')

