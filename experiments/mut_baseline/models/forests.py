
from dryadic.learning.pipelines import PresencePipe
from dryadic.learning.selection import SelectMeanVar

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


class Base(PresencePipe):
 
    tune_priors = (
        ('fit__max_features', tuple(10 ** np.linspace(-3, -2/3, 6))),
        ('fit__min_samples_leaf', (1, 2, 3, 4, 6, 8)),
        )

    feat_inst = SelectMeanVar(mean_perc=95, var_perc=95)
    norm_inst = StandardScaler()
    fit_inst = RandomForestClassifier(n_estimators=500,
                                      class_weight='balanced')

    def __init__(self):
        super().__init__([('feat', self.feat_inst), ('norm', self.norm_inst),
                          ('fit', self.fit_inst)])


class Few_trees(Base):

    tune_priors = (
        ('fit__max_features', tuple(10 ** np.linspace(-2.5, -0.4, 6))),
        ('fit__min_samples_leaf', (2, 3, 4, 5, 6, 7)),
        )

    fit_inst = RandomForestClassifier(n_estimators=100,
                                      class_weight='balanced')


class Many_trees(Base):

    fit_inst = RandomForestClassifier(n_estimators=2500,
                                      class_weight='balanced')

