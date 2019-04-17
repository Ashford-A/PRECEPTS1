
from dryadic.learning.pipelines import PresencePipe
from dryadic.learning.selection import SelectMeanVar

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier


class Base(PresencePipe):
 
    tune_priors = (
        ('fit__max_depth', (2, 3, 4, 5)),
        ('fit__min_samples_split', tuple(np.linspace(0.003, 0.051, 9))),
        )

    feat_inst = SelectMeanVar(mean_perc=95, var_perc=95)
    norm_inst = StandardScaler()
    fit_inst = GradientBoostingClassifier()

    def __init__(self):
        super().__init__([('feat', self.feat_inst), ('norm', self.norm_inst),
                          ('fit', self.fit_inst)])


class Many_estimators(Base):

    fit_inst = GradientBoostingClassifier(n_estimators=400)


class Depth_fixed(Base):

    tune_priors = (
        ('fit__learning_rate', (0.03, 0.05, 0.1, 0.15, 0.2, 0.3)),
        ('fit__min_samples_split', (1, 2, 5, 15, 40, 80)),
        )

    fit_inst = GradientBoostingClassifier(max_depth=4, n_estimators=150)

