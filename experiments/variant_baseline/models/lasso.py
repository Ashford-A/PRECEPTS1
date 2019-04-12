
from dryadic.learning.pipelines import PresencePipe
from dryadic.learning.selection import SelectMeanVar

import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score


class Base(PresencePipe):
    """Linear regression classifiers with the LASSO regularization penalty.

    Note that the `C` regularization strength parameter should have this same
    testing value grid in all cases where it is tuned over. The selected range
    of values reflects the finding that setting `C` to less than 0.01 doesn't
    appear to ever work in the context of predicting mutation status in any of
    the variants of LASSO regression given below, and also that past a certain
    point all large values of `C` will simply result in no regularization.
    """

    tune_priors = (
        ('fit__C', tuple(10 ** np.linspace(-2.25, 10, 36))),
        )

    feat_inst = SelectMeanVar(mean_perc=95, var_perc=95)
    norm_inst = StandardScaler()
    fit_inst = LogisticRegression(penalty='l1', max_iter=200,
                                  class_weight='balanced')

    def __init__(self):
        super().__init__([('feat', self.feat_inst), ('norm', self.norm_inst),
                          ('fit', self.fit_inst)])


class Meanvar(Base):
    """LASSO regression with tuning over feature selection thresholds.

    We fix a value of `C` that tends to work well across a wide variety of
    mutation prediction tasks, and then tune over different filter cutoffs for
    removing expression features with low mean or variance.
    """

    tune_priors = (
        ('feat__mean_perc', (50, 65, 75, 85, 90, 99)),
        ('feat__var_perc', (50, 65, 75, 85, 90, 99)),
        )

    feat_inst = SelectMeanVar()
    fit_inst = LogisticRegression(penalty='l1', max_iter=200, C=np.exp(1),
                                  class_weight='balanced')


class Norm_robust(Base):

    norm_inst = RobustScaler()


class Norm_minmax(Base):

    norm_inst = MinMaxScaler()


class Iter_short(Base):

    fit_inst = LogisticRegression(penalty='l1', max_iter=80,
                                  class_weight='balanced')


class Tune_aupr(Base):
    """LASSO regression with AUPR used as a measure of training accuracy.

    This classifier replaces AUC with AUPR to evaluate the performance of
    different values of `C` tested during tuning.

    TODO: Figure out why this method of tuning fails in most cases.
    """

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

