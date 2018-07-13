
from ....predict.pipelines import PresencePipe
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression


class Base(PresencePipe):

    tune_priors = (
        ('fit__C', tuple(10 ** np.linspace(-7.25, 4.25, 24))),
        )

    norm_inst = StandardScaler()
    fit_inst = LogisticRegression(penalty='l2', class_weight='balanced')

    def __init__(self):
        super().__init__([('norm', self.norm_inst), ('fit', self.fit_inst)])


class Norm_robust(Base):

    norm_inst = RobustScaler()

