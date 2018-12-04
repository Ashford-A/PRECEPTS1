
from dryadic.learning.pipelines import PresencePipe
from dryadic.learning.selection import SelectMeanVar

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier


class Base(PresencePipe):
 
    tune_priors = (
        ('fit__max_depth', (2, 3, 4, 5, 6, 7)),
        ('fit__min_samples_split', (0.005, 0.01, 0.02, 0.03, 0.04, 0.05)),
        )

    feat_inst = SelectMeanVar(mean_perc=95, var_perc=95)
    norm_inst = StandardScaler()
    fit_inst = GradientBoostingClassifier(n_estimators=50, learning_rate=0.2)

    def __init__(self):
        super().__init__([('feat', self.feat_inst), ('norm', self.norm_inst),
                          ('fit', self.fit_inst)])


class Depth_fixed(Base):

    tune_priors = (
        ('fit__learning_rate', (0.03, 0.07, 0.11, 0.17, 0.25, 0.35)),
        ('fit__min_samples_split', (0.006, 0.018, 0.028, 0.036, 0.46, 0.58)),
        )

    fit_inst = GradientBoostingClassifier(max_depth=4, n_estimators=150)

