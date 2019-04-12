
from dryadic.learning.pipelines import PresencePipe
from dryadic.learning.selection import SelectMeanVar

import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression


class Base(PresencePipe):
    """Linear regression classifiers with the ridge regularization penalty.

    Note that the `C` regularization strength parameter should have this same
    testing value grid in all cases where it is tuned over. This reflects that
    optimal values of `C` tend to always fall well within this selected range
    for observed mutation prediction tasks, and also that past a certain point
    all large values of `C` will result in no regularization.
    """


    tune_priors = (
        ('fit__C', tuple(10 ** np.linspace(-7.1, 3.4, 36))),
        )

    feat_inst = SelectMeanVar(mean_perc=95, var_perc=95)
    norm_inst = StandardScaler()
    fit_inst = LogisticRegression(penalty='l2', class_weight='balanced')

    def __init__(self):
        super().__init__([('feat', self.feat_inst), ('norm', self.norm_inst),
                          ('fit', self.fit_inst)])


class Meanvar(Base):
    """Ridge regression with tuning over feature selection thresholds.

    We fix a value of `C` that tends to work well across a wide variety of
    mutation prediction tasks, and then tune over different filter cutoffs for
    removing expression features with low mean or variance.
    """

    tune_priors = (
        ('feat__mean_perc', (100./3, 50, 75, 90, 98, 100)),
        ('feat__var_perc', (100./3, 50, 75, 90, 98, 100)),
        )

    feat_inst = SelectMeanVar()
    fit_inst = LogisticRegression(C=0.002,
                                  penalty='l2', class_weight='balanced')


class Norm_robust(Base):

    norm_inst = RobustScaler()


class Select_few(Base):

    feat_inst = SelectMeanVar(mean_perc=200./3, var_perc=200./3)

