"""Algorithms for use in predicting binary mutation states in cohorts."""

from dryadic.learning.classifiers import Base, LinearPipe
from dryadic.learning.selection import SelectMeanVar
import numpy as np
from sklearn.linear_model import LogisticRegression


class Lasso(Base, LinearPipe):

    feat_inst = SelectMeanVar(mean_perc=95, var_perc=100)

    tune_priors = (
        ('fit__C', tuple(10 ** np.linspace(-4, 3, 8))),
        )
    test_count = 8

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

