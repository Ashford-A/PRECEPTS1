
from ....predict.pipelines import PresencePipe
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


class Base(PresencePipe):
 
    tune_priors = (
        ('fit__max_features', tuple(10 ** np.linspace(-3.5, -1, 6))),
        ('fit__min_samples_leaf', (1, 3, 6, 10)),
        )

    norm_inst = StandardScaler()
    fit_inst = RandomForestClassifier(n_estimators=500,
                                      class_weight='balanced')

    def __init__(self):
        super().__init__([('norm', self.norm_inst), ('fit', self.fit_inst)])


class Many_trees(Base):

    fit_inst = RandomForestClassifier(n_estimators=2500,
                                      class_weight='balanced')

