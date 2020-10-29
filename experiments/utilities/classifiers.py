"""Algorithms for use in predicting binary mutation states in cohorts."""

from dryadic.learning.classifiers import Base, LinearPipe, Kernel, Trees
from dryadic.learning.selection import SelectMeanVar

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


class Lasso(Base, LinearPipe):

    feat_inst = SelectMeanVar(mean_perc=90, var_perc=100)

    tune_priors = (
        ('fit__C', tuple(10 ** np.linspace(-4, 3, 8))),
        )
    test_count = 8

    fit_inst = LogisticRegression(solver='liblinear', penalty='l1',
                                  max_iter=200, class_weight='balanced')


class Ridge(Base, LinearPipe):

    feat_inst = SelectMeanVar(mean_perc=90, var_perc=100)

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


class SVCrbf(Base, Kernel):

    feat_inst = SelectMeanVar(mean_perc=90, var_perc=100)

    tune_priors = (
        ('fit__C', tuple(10 ** np.linspace(-3, 4, 8))),
        )
    test_count = 8

    fit_inst = SVC(kernel='rbf', gamma='scale', probability=True,
                   cache_size=500, class_weight='balanced')


class Forests(Base, Trees):

    feat_inst = SelectMeanVar(mean_perc=90, var_perc=100)

    tune_priors = (
        ('fit__min_samples_leaf', (1, 2, 3, 4, 6, 8, 10, 15)),
        )
    test_count = 8
 
    fit_inst = RandomForestClassifier(n_estimators=5000,
                                      class_weight='balanced')

