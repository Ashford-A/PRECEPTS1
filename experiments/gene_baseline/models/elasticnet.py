
from ....predict.pipelines import PresencePipe
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import SGDClassifier


class Base(PresencePipe):

    tune_priors = (
        ('fit__alpha', tuple(10 ** np.linspace(-4, 1.25, 8))),
        ('fit__l1_ratio', (0.25, 0.5, 0.75)),
        )

    norm_inst = StandardScaler()
    fit_inst = SGDClassifier(loss='log', penalty='elasticnet',
                             max_iter=1000, class_weight='balanced')

    def __init__(self):
        super().__init__([('norm', self.norm_inst), ('fit', self.fit_inst)])


class Norm_robust(Base):

    norm_inst = RobustScaler()


class Ratio_skew(Base):

    tune_priors = (
        ('fit__alpha', tuple(10 ** np.linspace(-3.75, -0.25, 8))),
        ('fit__l1_ratio', (1.0/7, 1.0/5, 1.0/3)),
        )


class Ratio_fixed(Base):

    tune_priors = (
        ('fit__alpha', tuple(10 ** np.linspace(-3.45, 0, 24))),
        )

    fit_inst = SGDClassifier(
        l1_ratio=0.25, loss='log', penalty='elasticnet',
        max_iter=1000, class_weight='balanced'
        )

