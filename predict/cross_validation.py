
"""Applying cross-validation techniques in -omic prediction contexts.

This module contains utility functions that assist with cross-validation for
use in gauging the efficacy of algorithms applied to predict -omic phenotypes.

Author: Michal Grzadkowski <grzadkow@ohsu.edu>

"""

import numpy as np
import pandas as pd
import time

from collections import Sized, defaultdict
from functools import partial, reduce
from operator import or_
from itertools import product

import scipy.sparse as sp
from scipy.stats import rankdata

from sklearn.base import is_classifier, clone
from sklearn.utils import check_random_state
from sklearn.utils.fixes import MaskedArray
from sklearn.utils.validation import check_array, _num_samples
from sklearn.metrics.scorer import check_scoring
from sklearn.externals.joblib import Parallel, delayed, logger

from sklearn.model_selection import (
    StratifiedShuffleSplit, StratifiedKFold, RandomizedSearchCV)
from sklearn.model_selection._split import (
    _validate_shuffle_split, _approximate_mode, check_cv)
from sklearn.model_selection._validation import (
    _fit_and_predict, _score, _index_param_value)


def cross_val_predict_omic(estimator, X, y=None, groups=None,
                           force_test_samps=None,
                           cv_fold=4, cv_count=16, n_jobs=1, fit_params=None,
                           random_state=None, verbose=0):
    """Generates predicted mutation states for samples using internal
       cross-validation via repeated stratified K-fold sampling.
    """

    if (cv_count % cv_fold) != 0:
        raise ValueError("The number of folds should evenly divide the total"
                         "number of cross-validation splits.")

    if force_test_samps is None:
        train_samps_indx = np.arange(X.shape[0])
        test_samps_indx = np.array([], dtype=int)

    else:
        train_samps = set(X.index) - set(force_test_samps)
        train_samps_indx = X.index.get_indexer_for(train_samps)
        test_samps_indx = X.index.get_indexer_for(force_test_samps)

    #X, y = omic_indexable(X, y)
    #y = y.reshape(-1)

    cv_rep = int(cv_count / cv_fold)
    cv_iter = []

    if len(y.shape) > 1 and y.shape[1] > 1:
        y_use = np.apply_along_axis(lambda x: reduce(or_, x), 1, y)

    else:
        y_use = y.reshape(-1)

    for i in range(cv_rep):
        cv = StratifiedKFold(
            n_splits=cv_fold, shuffle=True,
            random_state=(random_state ** (i + 3)) % 12949671
            )
        
        cv_iter += [
            (train_samps_indx[train],
             np.append(train_samps_indx[test], test_samps_indx))
            for train, test in cv.split(X.iloc[train_samps_indx, :],
                                        y_use[train_samps_indx],
                                        groups)
            ]

    # for each split, fit on the training set and get predictions for
    # remaining cohort
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch='n_jobs')
    prediction_blocks = parallel(delayed(
        _omic_fit_and_predict)(clone(estimator), X, y,
                               train, test, verbose, fit_params)
        for train, test in cv_iter
        )

    # consolidates the predictions into an array
    pred_mat = [[] for _ in range(X.shape[0])]
    for i in range(cv_rep):
        predictions = np.concatenate(
            prediction_blocks[(i * cv_fold):((i + 1) * cv_fold)])

        test_indices = np.concatenate(
            [indices_i for _, indices_i
             in cv_iter[(i * cv_fold):((i + 1) * cv_fold)]]
            )

        for j in range(X.shape[0]):
            pred_mat[j] += predictions[test_indices == j].tolist()

    return pred_mat


def omic_indexable(omic, pheno):
    """Make arrays indexable for cross-validation.

    Checks consistent length, passes through None, and ensures that everything
    can be indexed by converting sparse matrices to csr and converting
    non-interable objects to arrays.

    Parameters
    ----------
    *iterables : lists, dataframes, arrays, sparse matrices
        List of objects to ensure sliceability.
    """

    if sp.issparse(omic):
        new_omic = omic.tocsr()

    elif hasattr(omic, "iloc"):
        new_omic = omic.values

    elif isinstance(omic, dict):
        new_omic = {lbl: np.array(x) for lbl, x in omic.items()}

    else:
        new_omic = np.array(omic)

    if sp.issparse(pheno):
        new_pheno = pheno.tocsr()

    elif hasattr(pheno, "iloc"):
        new_pheno = pheno.values

    elif isinstance(pheno, dict):
        new_pheno = {lbl: np.array(y) for lbl, y in pheno.items()}

    elif pheno is None:
        new_pheno = None

    else:
        new_pheno = np.array(pheno)

    check_consistent_omic_length(new_omic, new_pheno)

    return new_omic, new_pheno


def check_consistent_omic_length(omic, pheno):
    """Check that all arrays have consistent first dimensions.

    Checks whether all objects in arrays have the same shape or length.

    Parameters
    ----------
    *arrays : list or tuple of input objects.
        Objects that will be checked for consistent length.
    """

    if pheno is None:
        pass

    elif hasattr(omic, "shape") and hasattr(pheno, "shape"):
        if omic.shape[0] != pheno.shape[0]:
            raise ValueError(
                "Cannot use -omic dataset with {} samples to predict "
                "phenotypes measured on {} samples!".format(
                    omic.shape[0], pheno.shape[0])
                )

    # in the case of transfer learning, check that the input data and the
    # output labels have matching context labels
    elif isinstance(omic, dict) and isinstance(pheno, dict):
        if omic.keys() != pheno.keys():
            raise ValueError("-omic datasets and phenotypes must be "
                             "collected from the same contexts!")

        # also check that for each context, the input array has the same
        # number of samples as the output labels
        else:
            for lbl in omic:
                if omic[lbl].shape[0] != pheno[lbl].shape[0]:
                    raise ValueError(
                        "In context {}, cannot use -omic dataset with {} "
                        "samples to predict phenotypes measured on {} "
                        "samples!".format(lbl, omic[lbl].shape[0],
                                          pheno[lbl].shape[0])
                        )


def _omic_fit_and_predict(estimator, X, y, train, test, verbose, fit_params):

    fit_params = fit_params if fit_params is not None else {}
    fit_params = dict([(k, _index_param_value(X, v, train))
                      for k, v in fit_params.items()])

    X_train, y_train = _omic_safe_split(estimator, X, y, train)
    X_test, _ = _omic_safe_split(estimator, X, y, test, train)

    if y_train is None:
        estimator.fit(X_train, **fit_params)
    else:
        estimator.fit(X_train, y_train, **fit_params)

    return estimator.predict_omic(X_test)

def _omic_fit_and_score(estimator, X, y, scorer, train, test, verbose,
                        parameters, return_train_score=False,
                        return_parameters=False, return_n_test_samples=False,
                        return_times=False, error_score='raise'):
    """Fit estimator and compute scores for a given dataset split.

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    X : array-like of shape at least 2D
        The data to fit.

    y : array-like, optional, default: None
        The target variable to try to predict in the case of
        supervised learning.

    scorer : callable
        A scorer callable object / function with signature
        ``scorer(estimator, X, y)``.

    train : array-like, shape (n_train_samples,)
        Indices of training samples.

    test : array-like, shape (n_test_samples,)
        Indices of test samples.

    verbose : integer
        The verbosity level.

    error_score : 'raise' (default) or numeric
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error.

    parameters : dict or None
        Parameters to be set on the estimator.

    return_train_score : boolean, optional, default: False
        Compute and return score on training set.

    return_parameters : boolean, optional, default: False
        Return parameters that has been used for the estimator.

    Returns
    -------
    train_score : float, optional
        Score on training set, returned only if `return_train_score` is `True`.

    test_score : float
        Score on test set.

    n_test_samples : int
        Number of test samples.

    fit_time : float
        Time spent for fitting in seconds.

    score_time : float
        Time spent for scoring in seconds.

    parameters : dict or None, optional
        The parameters that have been evaluated.
    """
    if verbose > 1:
        if parameters is None:
            msg = ''
        else:
            msg = '%s' % (', '.join('%s=%s' % (k, v)
                          for k, v in parameters.items()))
        print("[CV] %s %s" % (msg, (64 - len(msg)) * '.'))

    start_time = time.time()
    X_train, y_train = _omic_safe_split(estimator, X, y, train)
    X_test, y_test = _omic_safe_split(estimator, X, y, test, train)
    est_params = estimator.get_params()

    for par, val in parameters.items():
        if par in est_params:
            estimator.set_params(**{par: val})

    for par in parameters.keys() & est_params.keys():
        del parameters[par]

    try:
        if y_train is None:
            estimator.fit(X_train, **parameters)
        else:
            estimator.fit(X_train, y_train, **parameters)

    except Exception as e:
        # Note fit time as time until error
        fit_time = time.time() - start_time
        score_time = 0.0
        if error_score == 'raise':
            raise
        elif isinstance(error_score, numbers.Number):
            test_score = error_score
            if return_train_score:
                train_score = error_score
            warnings.warn("Classifier fit failed. The score on this train-test"
                          " partition for these parameters will be set to %f. "
                          "Details: \n%r" % (error_score, e), FitFailedWarning)
        else:
            raise ValueError("error_score must be the string 'raise' or a"
                             " numeric value. (Hint: if using 'raise', please"
                             " make sure that it has been spelled correctly.)")

    else:
        fit_time = time.time() - start_time
        test_score = _score(estimator, X_test, y_test, scorer)
        score_time = time.time() - start_time - fit_time
        if return_train_score:
            train_score = _score(estimator, X_train, y_train, scorer)

    if verbose > 2:
        msg += ", score=%f" % test_score
    if verbose > 1:
        total_time = score_time + fit_time
        end_msg = "%s, total=%s" % (msg, logger.short_format_time(total_time))
        print("[CV] %s %s" % ((64 - len(end_msg)) * '.', end_msg))

    ret = [train_score, test_score] if return_train_score else [test_score]

    if return_n_test_samples:
        ret.append(_num_samples(X_test))
    if return_times:
        ret.extend([fit_time, score_time])
    if return_parameters:
        ret.append(parameters)
    return ret


def _omic_safe_split(estimator, X, y, indices, train_indices=None):
    """Create subset of dataset."""

    if hasattr(X, "iloc"):
        X_subset = X.iloc[indices]

    elif hasattr(X, "shape"):
        X_subset = X[indices]

    elif isinstance(X, dict):
        X_subset = {lbl: x.iloc[indices[lbl]] for lbl, x in X.items()}

    else:
        raise TypeError("Cannot safely split -omic dataset of unsupported "
                        "type {} !".format(type(X)))

    if y is not None:
        if hasattr(y, "shape"):
            y_subset = y[indices]

        elif isinstance(y, dict):
            y_subset = {lbl: y[lbl][indices[lbl]] for lbl in X}

        elif hasattr(y, "__getitem__"):
            y_subset = [y[i] for i in indices]

        elif hasattr(y, "__iter__"):
            y_subset = [y_val for i, y_val in enumerate(y) if i in indices]

        else:
            raise TypeError("Cannot safely split phenotype values of "
                            "unsupported type {} !".format(type(y)))

    else:
        y_subset = None

    return X_subset, y_subset


class OmicRandomizedCV(RandomizedSearchCV):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, X, y, groups=None, **tune_params):
        """Actual fitting,  performing the search over parameters."""

        # gets and copies the prediction algorithm
        estimator = self.estimator
        base_estimator = clone(self.estimator)

        # checks the methods used to produce cross validation splits and to
        # score the accuracy of the predictions
        cv = check_cv(self.cv, y, classifier=is_classifier(estimator))
        self.scorer_ = check_scoring(self.estimator, scoring=self.scoring)

        # finds how many splits of the data into training and testing sub-
        # cohorts will be used, and which parameter values will be tested on
        # each split
        n_splits = cv.get_n_splits(X, y, groups)
        candidate_params = list(self._get_param_iterator())
        n_candidates = len(candidate_params)

        # updates the parameter values to be tested with parameter values
        # specific to testing
        for i in range(len(candidate_params)):
            candidate_params[i].update(tune_params)

        if self.verbose > 0 and isinstance(tune_params, Sized):
            n_candidates = len(tune_params)
            print("Fitting {0} folds for each of {1} candidates, totalling"
                  " {2} fits".format(n_splits, n_candidates,
                                     n_candidates * n_splits))

        out = Parallel(
            n_jobs=self.n_jobs, pre_dispatch='n_jobs', verbose=self.verbose
            )(delayed(_omic_fit_and_score)(
                clone(base_estimator), X, y, self.scorer_,
                train, test, self.verbose, parameters,
                return_train_score=self.return_train_score,
                return_n_test_samples=True,
                return_times=True, return_parameters=True,
                error_score=self.error_score
                )
                for parameters, (train, test) in product(
                    candidate_params, cv.split(X, y, groups))
            )

        # if one choose to see train score, "out" will contain train score info
        if self.return_train_score:
            (train_scores, test_scores, test_sample_counts,
             fit_time, score_time, parameters) = zip(*out)
        else:
            (test_scores, test_sample_counts,
             fit_time, score_time, parameters) = zip(*out)

        results = dict()

        def _store(key_name, array, weights=None, splits=False, rank=False):
            """A small helper to store the scores/times to the cv_results_"""
            array = np.array(array, dtype=np.float64).reshape(n_candidates,
                                                              n_splits)
            if splits:
                for split_i in range(n_splits):
                    results["split%d_%s"
                            % (split_i, key_name)] = array[:, split_i]

            array_means = np.average(array, axis=1, weights=weights)
            results['mean_%s' % key_name] = array_means
            # Weighted std is not directly available in numpy
            array_stds = np.sqrt(np.average((array -
                                             array_means[:, np.newaxis]) ** 2,
                                            axis=1, weights=weights))
            results['std_%s' % key_name] = array_stds

            if rank:
                results["rank_%s" % key_name] = np.asarray(
                    rankdata(-array_means, method='min'), dtype=np.int32)

        # Computed the (weighted) mean and std for test scores alone
        # NOTE test_sample counts (weights) remain the same for all candidates
        test_sample_counts = np.array(test_sample_counts[:n_splits],
                                      dtype=np.int)

        _store('test_score', test_scores, splits=True, rank=True,
               weights=test_sample_counts if self.iid else None)
        if self.return_train_score:
            _store('train_score', train_scores, splits=True)
        _store('fit_time', fit_time)
        _store('score_time', score_time)

        # Use one MaskedArray and mask all the places where the param is not
        # applicable for that candidate. Use defaultdict as each candidate may
        # not contain all the params
        param_results = defaultdict(partial(MaskedArray,
                                            np.empty(n_candidates,),
                                            mask=True,
                                            dtype=object))
        for cand_i, params in enumerate(candidate_params):
            for name, value in params.items():
                # An all masked empty array gets created for the key
                # `"param_%s" % name` at the first occurence of `name`.
                # Setting the value at an index also unmasks that index
                param_results["param_%s" % name][cand_i] = value

        results.update(param_results)

        # saves the results of tuning the estimator as attributes
        results['params'] = candidate_params
        self.cv_results_ = results
        self.n_splits_ = n_splits

        return self


class OmicShuffleSplit(StratifiedShuffleSplit):
    """Generates splits of single or multiple cohorts into training and
       testing sets that are stratified according to the mutation vectors.
    """

    def _iter_indices(self, expr, omic=None, groups=None):
        """Generates indices of training/testing splits for use in
           stratified shuffle splitting of cohort data.
        """

        # with one domain and one variant to predict proceed with stratified
        # sampling, binning mutation values if they are continuous
        if hasattr(expr, 'shape') and hasattr(omic, 'shape'):

            if len(omic.shape) > 1 and omic.shape[1] > 1:
                omic_use = np.apply_along_axis(lambda x: reduce(or_, x),
                                               1, omic)

            elif len(np.unique(omic)) > 10:
                omic_use = omic > np.percentile(omic, 50)

            for train, test in super()._iter_indices(
                    X=expr, y=omic_use, groups=groups):

                yield train, test

        elif hasattr(omic, 'shape'):

            if len(np.unique(omic)) > 2:
                if len(omic.shape) == 1:
                    omic = omic > np.percentile(omic, 50)
                else:
                    if isinstance(omic, pd.DataFrame):
                        samp_mean = np.mean(omic.fillna(0.0), axis=1)
                    elif isinstance(omic, np.ndarray):
                        samp_mean = np.mean(np.nan_to_num(omic), axis=1)

                    omic = samp_mean > np.percentile(samp_mean, 50)

            for train, test in super()._iter_indices(
                    X=list(expr.values())[0], y=omic, groups=groups):

                yield train, test

        elif hasattr(expr, 'shape'):

            # gets info about input
            n_samples = _num_samples(expr)
            n_train, n_test = _validate_shuffle_split(
                n_samples, self.test_size, self.train_size)

            class_info = [np.unique(y, return_inverse=True) for y in omic]
            merged_classes = reduce(
                lambda x, y: x + y,
                [y_ind * 2 ** i for i, (_, y_ind) in enumerate(class_info)]
                )
            merged_counts = np.bincount(merged_classes)
            class_info = np.unique(merged_classes, return_inverse=True)

            new_counts = merged_counts.tolist()
            new_info = list(class_info)
            new_info[0] = new_info[0].tolist()

            remove_indx = []
            for i, count in enumerate(merged_counts):
                if count < 2 and i in new_info[0]:

                    remove_indx += [i]
                    cur_ind = merged_classes == i

                    if i > 0:
                        new_counts[i - 1] += new_counts[i]
                        rep_indx = new_info[0].index(i) - 1

                    else:
                        new_counts[i + 1] += new_counts[i]
                        rep_indx = new_info[0].index(i) + 1

                    merged_classes[cur_ind] = new_info[0][rep_indx]

            for i in remove_indx:
                new_info[0].remove(i)
            new_counts = np.array(new_counts)

            n_class = len(new_info[0])
            if n_train < n_class:
                raise ValueError('The train_size = %d should be greater or '
                                 'equal to the number of classes = %d'
                                 % (n_train, n_class))
            if n_test < n_class:
                raise ValueError('The test_size = %d should be greater or '
                                 'equal to the number of classes = %d'
                                 % (n_test, n_class))

            # generates random training and testing cohorts
            rng = check_random_state(self.random_state)
            for _ in range(self.n_splits):
                n_is = _approximate_mode(new_counts, n_train, rng)
                class_counts_remaining = new_counts - n_is
                t_is = _approximate_mode(class_counts_remaining, n_test, rng)

                train = []
                test = []

                for class_i in new_info[0]:
                    permutation = rng.permutation(new_counts[class_i])
                    perm_indices_class = np.where(
                        merged_classes == class_i)[0][permutation]

                    train.extend(perm_indices_class[:n_is[class_i]])
                    test.extend(
                        perm_indices_class[n_is[class_i]:(n_is[class_i]
                                                          + t_is[class_i])]
                        )

                    train = rng.permutation(train).tolist()
                    test = rng.permutation(test).tolist()

                yield train, test

        # otherwise, perform stratified sampling on each cohort separately
        else:

            # gets info about input
            n_samples = {lbl: _num_samples(X) for lbl, X in expr.items()}
            n_train_test = {
                lbl: _validate_shuffle_split(n_samps,
                                             self.test_size, self.train_size)
                for lbl, n_samps in n_samples.items()
                }

            class_info = {lbl: np.unique(y, return_inverse=True)
                          for lbl, y in omic.items()}
            n_classes = {lbl: classes.shape[0]
                         for lbl, (classes, _) in class_info.items()}
            classes_counts = {lbl: np.bincount(y_indices)
                              for lbl, (_, y_indices) in class_info.items()}

            # ensure we have enough samples in each class for stratification
            for lbl, (n_train, n_test) in n_train_test.items():
                if np.min(classes_counts[lbl]) < 2:
                    raise ValueError(
                        "The least populated phenotype class in {} has only "
                        "one member, which is too few. The minimum number of "
                        "groups for any phenotypic feature to predict cannot "
                        "be less than two.".format(lbl)
                        )

                if n_train < n_classes[lbl]:
                    raise ValueError(
                        "The number of training samples ({}) should be "
                        "greater or equal to the number of "
                        "phenotypes ({})".format(n_train, n_classes[lbl])
                        )

                if n_test < n_classes[lbl]:
                    raise ValueError(
                        "The number of testing samples ({}) should be "
                        "greater or equal to the number of "
                        "phenotypes ({})".format(n_test, n_classes[lbl])
                        )

            # generates random training and testing cohorts
            rng = check_random_state(self.random_state)
            for _ in range(self.n_splits):
                n_is = {lbl: _approximate_mode(classes_counts[lbl],
                                               n_train_test[lbl][0], rng)
                        for lbl in expr}

                classes_counts_left = {lbl: classes_counts[lbl] - n_is[lbl]
                                       for lbl in expr}
                t_is = {lbl: _approximate_mode(classes_counts_left[lbl],
                                               n_train_test[lbl][1], rng)
                        for lbl in expr}

                train = {lbl: [] for lbl in expr}
                test = {lbl: [] for lbl in expr}

                for lbl, (classes, _) in class_info.items():
                    for i, class_i in enumerate(classes):
                        permutation = rng.permutation(classes_counts[lbl][i])

                        perm_indices_class_i = np.where(
                            (omic[lbl] == class_i))[0][permutation]
                        train[lbl].extend(perm_indices_class_i[:n_is[lbl][i]])

                        test[lbl].extend(
                            perm_indices_class_i[n_is[lbl][i]:n_is[lbl][i]
                                                 + t_is[lbl][i]]
                            )

                    train[lbl] = rng.permutation(train[lbl])
                    test[lbl] = rng.permutation(test[lbl])

                yield train, test

    def split(self, expr, omic=None, groups=None):
        """Gets the training/testing splits for a cohort."""

        if isinstance(omic, np.ndarray):
            omic = check_array(omic, ensure_2d=False, dtype=None)

        elif isinstance(omic, pd.DataFrame):
            omic = check_array(omic.values, ensure_2d=False, dtype=None,
                               force_all_finite=False)

        elif isinstance(omic, list):
            omic = [check_array(y, ensure_2d=False, dtype=None) for y in omic]

        elif isinstance(omic, dict):
            omic = {lbl: check_array(y, ensure_2d=False, dtype=None)
                    for lbl, y in omic.items()}

        else:
            raise ValueError("Output values must be either a list of features"
                             "for a set of tasks or an numpy array of"
                             "features for a single task!")

        expr, omic = omic_indexable(expr, omic)
        return self._iter_indices(expr, omic)

