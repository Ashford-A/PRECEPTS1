
from dryadic.learning.classifiers import Base, LinearPipe, Kernel, Trees
from dryadic.learning.selection import SelectMeanVar
from dryadic.learning.scalers import center_scale

import numpy as np
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


class Lasso(Base, LinearPipe):

    tune_priors = (
        ('fit__C', tuple(10 ** np.linspace(-3.1, 6.1, 185))),
        )

    fit_inst = LogisticRegression(solver='liblinear', penalty='l1',
                                  max_iter=200, class_weight='balanced')


class Ridge(Base, LinearPipe):

    feat_inst = SelectMeanVar(mean_perc=95, var_perc=100)

    tune_priors = (
        ('fit__C', tuple(10 ** np.linspace(-7, 0, 8))),
        )
    test_count = 8

    fit_inst = LogisticRegression(solver='liblinear', penalty='l2',
                                  max_iter=200, class_weight='balanced')


class RidgeMoreTune(Ridge):

    tune_priors = (
        ('fit__C', tuple(10 ** np.linspace(-8.2, 4.2, 32))),
        )
    test_count = 32


class RidgeFlat(Ridge):

    tune_priors = (
        ('fit__C', tuple(10 ** np.linspace(-2, 7, 181))),
        )

    norm_inst2 = Normalizer()

    def __init__(self):
        super(Base, self).__init__([
            ('feat', self.feat_inst), ('norm', self.norm_inst),
            ('norm2', self.norm_inst2), ('fit', self.fit_inst)
            ])


class RidgeWhite(Ridge):

    norm_inst2 = center_scale

    def __init__(self):
        super(Base, self).__init__([
            ('feat', self.feat_inst), ('norm', self.norm_inst),
            ('norm2', self.norm_inst2), ('fit', self.fit_inst)
            ])


class Elastic(Base, LinearPipe):

    tune_priors = (
        ('fit__alpha', tuple(10 ** np.linspace(-7.25, 1.5, 36))),
        ('fit__l1_ratio', tuple(np.linspace(0.17, 0.83, 23))),
        )
 
    fit_inst = SGDClassifier(loss='log', penalty='elasticnet',
                             max_iter=1000, class_weight='balanced')


class SVCrbf(Base, Kernel):
 
    tune_priors = (
        ('fit__C', tuple(10 ** np.linspace(-3, 4, 8))),
        )
    test_count = 8

    fit_inst = SVC(kernel='rbf', gamma='scale', probability=True,
                   cache_size=500, class_weight='balanced')


class Forests(Base, Trees):
 
    tune_priors = (
        ('fit__min_samples_leaf', (1, 2, 3, 4, 6, 8, 10, 15)),
        )
    test_count = 8

    fit_inst = RandomForestClassifier(n_estimators=5000,
                                      class_weight='balanced')

