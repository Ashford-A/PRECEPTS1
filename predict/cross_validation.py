
"""
Hetman (Heterogeneity Manifold)
Prediction of mutation sub-types using expression data.
This file contains utility functions for use in performing cross-validation
on multi-domain and multi-task prediction tasks.
"""

# Author: Michal Grzadkowski <grzadkow@ohsu.edu>

import numpy as np
import scipy.sparse as sp
import time

from sklearn.utils.validation import check_array, _num_samples
from sklearn.utils.fixes import bincount

from sklearn.model_selection import (
    StratifiedShuffleSplit, StratifiedKFold)
from sklearn.model_selection._split import (
    _validate_shuffle_split, _approximate_mode)
from sklearn.model_selection._validation import _fit_and_predict, _score
from sklearn.model_selection import RandomizedSearchCV

from collections import Sized, defaultdict
from functools import partial, reduce

from sklearn.base import is_classifier, clone
from sklearn.model_selection._split import check_cv
from sklearn.model_selection._validation import (
    _fit_and_score, _index_param_value)
from sklearn.externals.joblib import Parallel, delayed, logger
from sklearn.utils import check_random_state, indexable
from sklearn.utils.fixes import rankdata
from sklearn.utils.fixes import MaskedArray
from sklearn.metrics.scorer import check_scoring


def cross_val_predict_mut(estimator, X, y=None, groups=None,
                          exclude_samps=None, cv_fold=4, cv_count=16,
                          n_jobs=1, verbose=0, fit_params=None,
                          pre_dispatch='2*n_jobs', random_state=None):
    """Generates predicted mutation states for samples using internal
       cross-validation via repeated stratified K-fold sampling.
    """

    # gets the number of K-fold repeats
    if (cv_count % cv_fold) != 0:
        raise ValueError("The number of folds should evenly divide the total"
                         "number of cross-validation splits.")
    cv_rep = int(cv_count / cv_fold)

    # checks that the given estimator can predict continuous mutation states
    if not callable(getattr(estimator, 'predict_proba')):
        raise AttributeError('predict_proba not implemented in estimator')

    # gets absolute indices for samples to train and test over
    X, y, groups = indexable(X, y, groups)
    if exclude_samps is None:
        exclude_samps = []
    else:
        exclude_samps = list(set(exclude_samps) - set(X.index[y]))
    use_samps = list(set(X.index) - set(exclude_samps))
    use_samps_indx = X.index.get_indexer_for(use_samps)
    ex_samps_indx = X.index.get_indexer_for(exclude_samps)

    # generates the training/prediction splits
    cv_iter = []
    for i in range(cv_rep):
        cv = StratifiedKFold(n_splits=cv_fold, shuffle=True,
                             random_state=(random_state * i) % 12949671)
        cv_iter += [
            (use_samps_indx[train],
             np.append(use_samps_indx[test], ex_samps_indx))
            for train, test in cv.split(X.ix[use_samps_indx, :],
                                        np.array(y)[use_samps_indx],
                                        groups)
            ]

    # for each split, fit on the training set and get predictions for
    # remaining cohort
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
                        pre_dispatch=pre_dispatch)
    prediction_blocks = parallel(delayed(_fit_and_predict)(
        clone(estimator), X, y,
        train, test, verbose, fit_params, 'predict_proba')
                                 for train, test in cv_iter)

    # consolidates the predictions into an array
    pred_mat = [[] for _ in range(X.shape[0])]
    for i in range(cv_rep):
        predictions = np.concatenate(
            [pred_block_i for pred_block_i, _
             in prediction_blocks[(i * cv_fold):((i + 1) * cv_fold)]])
        test_indices = np.concatenate(
            [indices_i for _, indices_i
             in prediction_blocks[(i * cv_fold):((i + 1) * cv_fold)]]
            )

        for j in range(X.shape[0]):
            pred_mat[j] += list(predictions[test_indices == j, 1])

    return pred_mat


def mut_indexable(expr, mut):
    """Make arrays indexable for cross-validation.

    Checks consistent length, passes through None, and ensures that everything
    can be indexed by converting sparse matrices to csr and converting
    non-interable objects to arrays.

    Parameters
    ----------
    *iterables : lists, dataframes, arrays, sparse matrices
        List of objects to ensure sliceability.
    """

    if sp.issparse(expr):
        new_expr = expr.tocsr()
    elif hasattr(expr, "__getitem__") or hasattr(expr, "iloc"):
        new_expr = expr
    elif expr is None:
        new_expr = None
    else:
        new_expr = np.array(expr)

    if sp.issparse(mut):
        new_mut = mut.tocsr()
    elif hasattr(mut, "__getitem__") or hasattr(mut, "iloc"):
        new_mut = mut
    elif mut is None:
        new_mut = None
    else:
        new_mut = np.array(mut)

    # check_consistent_mut_length(new_expr, new_mut)
    return new_expr, new_mut


def check_consistent_mut_length(expr, mut):
    """Check that all arrays have consistent first dimensions.

    Checks whether all objects in arrays have the same shape or length.

    Parameters
    ----------
    *arrays : list or tuple of input objects.
        Objects that will be checked for consistent length.
    """

    expr_lengths = _num_samples(expr)
    mut_lengths = _num_samples(mut)

    if len(np.unique(expr_lengths)):
        pass

    if len(np.unique(mut_lengths)):
        pass

    if expr_lengths != mut_lengths:
        pass

    #if len(uniques) > 1:
    #    raise ValueError("Found input variables with inconsistent numbers of"
    #                     " samples: %r" % [int(l) for l in mut_lengths])


def _mut_fit_and_score(estimator, X, y, scorer, train, test, verbose,
                       parameters, fit_params, return_train_score=False,
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

    fit_params : dict or None
        Parameters that will be passed to ``estimator.fit``.

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

    # Adjust length of sample weights
    fit_params = fit_params if fit_params is not None else {}
    fit_params = dict([(k, _index_param_value(X, v, train))
                      for k, v in fit_params.items()])

    if parameters is not None:
        estimator.set_params(**parameters)

    start_time = time.time()

    X_train, y_train = _mut_safe_split(estimator, X, y, train)
    X_test, y_test = _mut_safe_split(estimator, X, y, test, train)

    try:
        if y_train is None:
            estimator.fit(X_train, **fit_params)
        else:
            estimator.fit(X_train, y_train, **fit_params)

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


def _mut_safe_split(estimator, X, y, indices, train_indices=None):
    """Create subset of dataset and properly handle kernels."""
    from sklearn.gaussian_process.kernels import Kernel as GPKernel

    if (hasattr(estimator, 'kernel') and callable(estimator.kernel) and
            not isinstance(estimator.kernel, GPKernel)):
        # cannot compute the kernel values with custom function
        raise ValueError("Cannot use a custom kernel function. "
                         "Precompute the kernel matrix instead.")

    if not hasattr(X, "shape"):
        if getattr(estimator, "_pairwise", False):
            raise ValueError("Precomputed kernels or affinity matrices have "
                             "to be passed as arrays or sparse matrices.")
        X_subset = [X[index] for index in indices]
    else:
        if getattr(estimator, "_pairwise", False):
            # X is a precomputed square kernel matrix
            if X.shape[0] != X.shape[1]:
                raise ValueError("X should be a square kernel matrix")
            if train_indices is None:
                X_subset = X[np.ix_(indices, indices)]
            else:
                X_subset = X[np.ix_(indices, train_indices)]
        else:
            X_subset = mut_safe_indexing(X, indices)

    if y is not None:
        y_subset = mut_safe_indexing(y, indices)
    else:
        y_subset = None

    return X_subset, y_subset


def mut_safe_indexing(X, indices):
    """Return items or rows from X using indices.

    Allows simple indexing of lists or arrays.

    Parameters
    ----------
    X : array-like, sparse-matrix, list.
        Data from which to sample rows or items.

    indices : array-like, list
        Indices according to which X will be subsampled.
    """
    if hasattr(X, "iloc"):
        # Pandas Dataframes and Series
        #try:
        #    return X.iloc[indices]
        #except ValueError:
            # Cython typed memoryviews internally used in pandas do not support
            # readonly buffers.
        #    warnings.warn("Copying input dataframe for slicing.",
        #                  DataConversionWarning)
            return X.copy().iloc[indices]
    elif hasattr(X, "shape"):
        if hasattr(X, 'take') and (hasattr(indices, 'dtype') and
                                   indices.dtype.kind == 'i'):
            # This is often substantially faster than X[indices]
            return X.take(indices, axis=0)
        else:
            return X[indices]
    else:
        return [[x[idx] for x in X] for idx in indices]


class MutRandomizedCV(RandomizedSearchCV):

    def __init__(self, **kwargs):
        super(MutRandomizedCV, self).__init__(**kwargs)

    def _fit(self, X, y, groups, parameter_iterable):
        """Actual fitting,  performing the search over parameters."""

        estimator = self.estimator
        cv = check_cv(self.cv, y, classifier=is_classifier(estimator))
        self.scorer_ = check_scoring(self.estimator, scoring=self.scoring)

        # X, y, groups = mut_indexable(X, y, groups)
        n_splits = cv.get_n_splits(X, y, groups)
        if self.verbose > 0 and isinstance(parameter_iterable, Sized):
            n_candidates = len(parameter_iterable)
            print("Fitting {0} folds for each of {1} candidates, totalling"
                  " {2} fits".format(n_splits, n_candidates,
                                     n_candidates * n_splits))

        base_estimator = clone(self.estimator)
        pre_dispatch = self.pre_dispatch

        cv_iter = list(cv.split(X, y, groups))
        out = Parallel(
            n_jobs=self.n_jobs, verbose=self.verbose,
            pre_dispatch=pre_dispatch
        )(delayed(_mut_fit_and_score)(clone(base_estimator), X, y, self.scorer_,
                                      train, test, self.verbose, parameters,
                                      fit_params=self.fit_params,
                                      return_train_score=self.return_train_score,
                                      return_n_test_samples=True,
                                      return_times=True, return_parameters=True,
                                      error_score=self.error_score)
          for parameters in parameter_iterable
          for train, test in cv_iter)

        # if one choose to see train score, "out" will contain train score info
        if self.return_train_score:
            (train_scores, test_scores, test_sample_counts,
             fit_time, score_time, parameters) = zip(*out)
        else:
            (test_scores, test_sample_counts,
             fit_time, score_time, parameters) = zip(*out)

        candidate_params = parameters[::n_splits]
        n_candidates = len(candidate_params)

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

        best_index = np.flatnonzero(results["rank_test_score"] == 1)[0]
        best_parameters = candidate_params[best_index]

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

        # Store a list of param dicts at the key 'params'
        results['params'] = candidate_params

        self.cv_results_ = results
        self.best_index_ = best_index
        self.n_splits_ = n_splits

        if self.refit:
            # fit the best estimator using the entire dataset
            # clone first to work around broken estimators
            best_estimator = clone(base_estimator).set_params(
                **best_parameters)
            if y is not None:
                best_estimator.fit(X, y, **self.fit_params)
            else:
                best_estimator.fit(X, **self.fit_params)
            self.best_estimator_ = best_estimator

        return self


class MutShuffleSplit(StratifiedShuffleSplit):
    """Generates splits of single or multiple cohorts into training and
       testing sets that are stratified according to the mutation vectors.
    """

    def __init__(self,
                 n_splits=10, test_size=0.1, train_size=None,
                 random_state=None):
        super(MutShuffleSplit, self).__init__(
            n_splits, test_size, train_size, random_state)

    def _iter_indices(self, expr, mut=None, groups=None):
        """Generates indices of training/testing splits for use in
           stratified shuffle splitting of cohort data.
        """

        # with one domain and one variant to predict proceed with stratified
        # sampling, binning mutation values if they are continuous
        if hasattr(expr, 'shape') and hasattr(mut, 'shape'):

            if len(np.unique(mut)) > 2:
                mut = mut > np.percentile(mut, 50)
            for train, test in super(MutShuffleSplit, self)._iter_indices(
                    X=expr, y=mut, groups=groups):

                yield train, test

        # with one domain and multiple variants
        elif hasattr(expr, 'shape'):

            # gets info about input
            n_samples = _num_samples(expr)
            mut = [check_array(y, ensure_2d=False, dtype=None)
                   for y in mut]
            n_train, n_test = _validate_shuffle_split(
                n_samples, self.test_size, self.train_size)

            class_info = [np.unique(y, return_inverse=True) for y in mut]
            merged_classes = reduce(
                lambda x, y: x + y,
                [y_ind * 2 ** i for i, (_, y_ind) in enumerate(class_info)]
                )
            merged_counts = bincount(merged_classes)
            class_info = np.unique(merged_classes, return_inverse=True)

            new_counts = merged_counts.tolist()
            new_info = list(class_info)
            new_info[0] = new_info[0].tolist()

            for i, count in enumerate(merged_counts):
                if count < 2:
                    cur_ind = merged_classes == new_info[0][i]
                    if i > 0:
                        new_counts[i - 1] += new_counts[i]
                        merged_classes[cur_ind] = new_info[0][i-1]
                    else:
                        new_counts[i + 1] += new_counts[i]
                        merged_classes[cur_ind] = new_info[0][i+1]

                    new_info[0] = new_info[0][:i] + new_info[0][(i+1):]
                    new_counts = new_counts[:i] + new_counts[(i+1):]
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
                for i, class_i in enumerate(new_info[0]):
                    permutation = rng.permutation(new_counts[i])

                    perm_indices_class = np.where(
                        merged_classes == class_i)[0][permutation]
                    train.extend(perm_indices_class[:n_is[i]])
                    test.extend(perm_indices_class[n_is[i]:n_is[i] + t_is[i]])

                    train = rng.permutation(train).tolist()
                    test = rng.permutation(test).tolist()

                yield train, test

        # otherwise, perform stratified sampling on each cohort separately
        else:

            # gets info about input
            n_samples = [_num_samples(X) for X in expr]
            mut = [check_array(y, ensure_2d=False, dtype=None)
                   for y in mut]
            n_train_test = [
                _validate_shuffle_split(n_samps,
                                        self.test_size, self.train_size)
                for n_samps in n_samples]
            class_info = [np.unique(y, return_inverse=True) for y in mut]
            n_classes = [classes.shape[0] for classes, _ in class_info]
            classes_counts = [bincount(y_indices)
                              for _, y_indices in class_info]

            # ensure we have enough samples in each class for stratification
            for i, (n_train, n_test) in enumerate(n_train_test):
                if np.min(classes_counts[i]) < 2:
                    raise ValueError(
                        "The least populated class in y has only 1 "
                        "member, which is too few. The minimum "
                        "number of groups for any class cannot "
                        "be less than 2.")

                if n_train < n_classes[i]:
                    raise ValueError(
                        'The train_size = %d should be greater or '
                        'equal to the number of classes = %d'
                        % (n_train, n_classes[i]))

                if n_test < n_classes[i]:
                    raise ValueError(
                        'The test_size = %d should be greater or '
                        'equal to the number of classes = %d'
                        % (n_test, n_classes[i]))

            # generates random training and testing cohorts
            rng = check_random_state(self.random_state)
            for _ in range(self.n_splits):
                n_is = [_approximate_mode(class_counts, n_train, rng)
                        for class_counts, (n_train, _)
                        in zip(classes_counts, n_train_test)]
                classes_counts_remaining = [class_counts - n_i
                                            for class_counts, n_i
                                            in zip(classes_counts, n_is)]
                t_is = [_approximate_mode(class_counts_remaining, n_test, rng)
                        for class_counts_remaining, (_, n_test)
                        in zip(classes_counts_remaining, n_train_test)]

                train = [[] for _ in expr]
                test = [[] for _ in expr]

                for i, (classes, _) in enumerate(class_info):
                    for j, class_j in enumerate(classes):
                        permutation = rng.permutation(classes_counts[i][j])
                        perm_indices_class_j = np.where(
                            (mut[i] == class_j))[0][permutation]
                        train[i].extend(perm_indices_class_j[:n_is[i][j]])
                        test[i].extend(
                            perm_indices_class_j[n_is[i][j]:n_is[i][j]
                                                            + t_is[i][j]])
                    train[i] = rng.permutation(train[i])
                    test[i] = rng.permutation(test[i])

                yield train, test

    def split(self, expr, mut=None, groups=None):
        """Gets the training/testing splits for a cohort."""

        if not hasattr(expr, 'shape') or not hasattr(mut, 'shape'):
            mut = [check_array(y, ensure_2d=False, dtype=None) for y in mut]

        else:
            mut = check_array(mut, ensure_2d=False, dtype=None)

        expr, mut = mut_indexable(expr, mut)
        return self._iter_indices(expr, mut)

