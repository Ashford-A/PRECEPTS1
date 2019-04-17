
from dryadic.learning.pipelines import PresencePipe
from dryadic.learning.selection import SelectMeanVar

import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import SGDClassifier


class Base(PresencePipe):
    """Linear regression classifiers with elastic net regularization.

    The `Base` model takes a symmetric sample over `l1_ratio` regardless of
    the performance of the classifier at each value of this parameter in order
    for us to gain a better understanding of how mixing the LASSO and ridge
    regularization penalties affects classifier behaviour.

    Models that implement tuning grids that favour better-performing values of
    `l1_ratio` are implemented below.
    """

    tune_priors = (
        ('fit__alpha', tuple(10 ** np.linspace(-4, -1/3, 12))),
        ('fit__l1_ratio', (0.25, 0.5, 0.75)),
        )

    feat_inst = SelectMeanVar(mean_perc=95, var_perc=95)
    norm_inst = StandardScaler()
    fit_inst = SGDClassifier(loss='log', penalty='elasticnet',
                             max_iter=1000, class_weight='balanced')

    def __init__(self):
        super().__init__([('feat', self.feat_inst), ('norm', self.norm_inst),
                          ('fit', self.fit_inst)])


class Meanvar(Base):
    """Elastic net regression with tuning over feature selection thresholds.

    We fix values of `alpha` and `l1_ratio` that tend to work well across a
    wide variety of mutation prediction tasks, and then tune over different
    filter cutoffs for removing expression features with low mean or variance.
    """

    tune_priors = (
        ('feat__mean_perc', (50, 65, 75, 85, 90, 99)),
        ('feat__var_perc', (50, 65, 75, 85, 90, 99)),
        )

    feat_inst = SelectMeanVar()
    fit_inst = SGDClassifier(loss='log', penalty='elasticnet', max_iter=1000,
                             l1_ratio=0.5, alpha=0.01,
                             class_weight='balanced')


class Norm_robust(Base):

    tune_priors = (
        ('fit__alpha', tuple(10 ** np.linspace(-6.7, -0.1, 12))),
        ('fit__l1_ratio', (0.25, 0.5, 0.75)),
        )

    norm_inst = RobustScaler()


class Ratio_skew(Base):
    """Elastic net regression with an asymmetric tuning grid over l1 ratio.

    Based on the observation that the elastic net model seems to perform
    better on mutation tasks when smaller values of `l1_ratio` are used, here
    we select values for this parameter that are skewed towards the lower end
    of the possible range.
    """

    tune_priors = (
        ('fit__alpha', tuple(10 ** np.linspace(-3.5, -0.3, 9))),
        ('fit__l1_ratio', (0.15, 0.3, 0.45, 0.6)),
        )


class Ratio_fixed(Base):

    tune_priors = (
        ('fit__alpha', tuple(10 ** np.linspace(-4.4, -0.2, 36))),
        )

    fit_inst = SGDClassifier(l1_ratio=0.4, loss='log', penalty='elasticnet',
                             max_iter=1000, class_weight='balanced')


class Ratio_low(Base):

    tune_priors = (
        ('fit__alpha', tuple(10 ** np.linspace(-7.5, 1.25, 36))),
        )

    fit_inst = SGDClassifier(l1_ratio=1./11, loss='log', penalty='elasticnet',
                             max_iter=1000, class_weight='balanced')


class Iter_short(Base):

    fit_inst = SGDClassifier(loss='log', penalty='elasticnet',
                             max_iter=100, class_weight='balanced')

