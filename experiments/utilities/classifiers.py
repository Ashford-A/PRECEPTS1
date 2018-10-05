
"""Standard classification algorithms to use for predicting -omic features.

This module contains a set of well-known classifiers that can be applied to a
wide variety of tasks involving the prediction of a binary -omic feature such
as the mutation status of a gene from a series of continuous -omic features
such as expression levels.

Each of these algorithms has been assigned a set of hyper-parameter values to
tune over, which have been chosen based on observations from how these
algorithms behave when applied to various -omic prediction tasks (see eg.
:func:`HetMan.experiments.mut_baseline.plot_model.plot_tuning_mtype` or
:module:`HetMan.experiments.subvariant_isolate.plot_tuning`). More details
about how these values were chosen for each classifier are provided below. In
general, we seek to choose a range of values centered around those that tend
to be selected as optimal after a tuning step, leaving a buffer at the edges
to allow for the possibility that predicition tasks we have not seen yet would
prefer a different set of values.

"""

from dryadic.learning.pipelines import PresencePipe
from dryadic.learning.selection import SelectMeanVar
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


class Base(PresencePipe):
    """An abstract class for the set of standard classifiers.

    The classifiers in this module are designed to use all available -omic
    features save those with very low expression in order to get the fullest
    possible picture of the features involved in a prediction signature for a
    given task. We save more sophisticated methods of feature selection for
    cases where we care more about classification performance, where we want
    to compare different priors about gene neighbourhoods to one another, etc.

    """

    feat_inst = SelectMeanVar(mean_perc=98, var_perc=100)
    norm_inst = StandardScaler()

    def __init__(self):
        super().__init__([('feat', self.feat_inst), ('norm', self.norm_inst),
                          ('fit', self.fit_inst)])

    def _feat_norm(self, X):
        return self.named_steps['norm'].transform(
            self.named_steps['feat'].transform(X))


class Linear(object):
    """An abstract class for classifiers assigning linear weights to features.

    """

    def calc_pred_labels(self, X):
        pred_lbls = np.dot(self._feat_norm(X),
                           self.named_steps['fit'].coef_.transpose())
        pred_lbls += self.named_steps['fit'].intercept_

        return pred_lbls.reshape(-1)


class Kernel(object):
    """An abstract class for non-linear support vector machines.

    """

    def calc_pred_labels(self, X):
        return self.decision_function(X).reshape(-1)


class Trees(object):
    """An abstract class for ensembles of decision trees.

    """

    def calc_pred_labels(self, X):
        return self.named_steps['fit'].predict_proba(
            self._feat_norm(X))[:, 1] * 2. - 1.


class Lasso(Base, Linear):
    """A linear regressor using logistic loss and lasso regularization.

    In the context of predicting mutation status using the standard set of
    ~16000 gene-based expression features, observed `fit__C` of above ~5.5
    imply assigning non-zero values to all available features in the final
    model. Tuning over values of this regularization strength parameter
    greater than this threshold thus appears to be redundant, even if its
    tuned values tend to approach the upper bound of the values that were
    tested in experimental settings.

    """

    tune_priors = (
        ('fit__C', tuple(10 ** np.linspace(-3.1, 6.1, 185))),
        )

    fit_inst = LogisticRegression(penalty='l1', max_iter=200,
                                  class_weight='balanced')


class Ridge(Base, Linear):
    """A linear regressor using logistic loss and ridge regularization.

    In the context of predicting mutation status using the standard set of
    ~16000 gene-based expression features, observed `fit__C` values close to
    the lower and upper edges of the tuning range given below imply all zero
    values and no regularization respectively. As with the :class:`Lasso`
    classifier above, tuning over values of `C` outside of this range can thus
    be assumed to be redundant.

    """

    tune_priors = (
        ('fit__C', tuple(10 ** np.linspace(-6.9, 6.9, 139))),
        )

    fit_inst = LogisticRegression(penalty='l2', class_weight='balanced')


class Elastic(Base, Linear):
    """A linear regressor using logistic loss and elastic net regularization.

    This classifier tends to prefer lower values for `l1_ratio` in the context
    of predicting mutation status from expression status, which is concordant
    with the superior performance of the ridge classifier in the same
    contexts.

    While tuned `fit__alpha` values tend to consistently fall between 1e-4 and
    1.0 across mutation prediction tasks, especially for high tuned values of
    `l1_ratio`, we consistently observe outlier tasks for which much lower
    `alpha` values are chosen, and we thus extend our range of values to tune
    over in an attempt to accomodate these cases.

    """

    tune_priors = (
        ('fit__alpha', tuple(10 ** np.linspace(-7.25, 1.5, 36))),
        ('fit__l1_ratio', tuple(np.linspace(0.17, 0.83, 23))),
        )
 
    fit_inst = SGDClassifier(loss='log', penalty='elasticnet',
                             max_iter=1000, class_weight='balanced')


class SVClinear(Base, Linear):
    """A linear classifier implemented using a support vector machine.

    While tuned `fit__C` values tend to fall between 1e-6 and 1e-3 across
    mutation prediction tasks, a subset of tasks tend to prefer higher values
    of this parameter, which most likely reflects an indifference to the
    strength of the regularization penalty. Nevertheless, we tune over at
    least some higher values of `C` to try and accomodate these cases.

    """

    tune_priors = (
        ('fit__C', tuple(10 ** np.linspace(-6.8, 2.8, 161))),
        )
 
    fit_inst = SVC(kernel='linear', probability=True,
                   cache_size=500, class_weight='balanced')


class SVCquad(Base, Kernel):
    """A quadratic classifier implemented using a support vector machine.

    Note that many -omic classification tasks appear to be insensitive to the
    value of `C`, and so we put a reasonable upper limit on its tuned values
    despite the fact that in many cases the final selected value will be at
    this upper limit.

    The `gamma` parameter appears to have a trade-off with `C`, with lower
    values of `gamma` leading to a preference for higher values of `C`. We
    thus constrain the values of `gamma` we tune over, especially at the lower
    bound. This is based on the assumption that tasks that would have chosen a
    very small tuned value for `gamma` can find a larger tuned value for
    `gamma` with a correspondingly lower value for `C` without any effect on
    classification behaviour.

    """

    tune_priors = (
        ('fit__C', tuple(10 ** np.linspace(-3, 7, 51))),
        ('fit__gamma', tuple(10 ** np.linspace(-8.75, -1.25, 11))),
        )
 
    fit_inst = SVC(kernel='poly', degree=2, coef0=1, probability=True,
                   cache_size=500, class_weight='balanced')


class SVCpoly(Base, Kernel):
    """A polynomial classifier implemented using a support vector machine.

    Given the difficulty of tuning over every possible useful value of the
    hyper-parameters available for this classifier, we restrict ourselves to
    "reasonable" values, and ignore the fact the many observed classification
    tasks prefer combinations of values at the edges of the tuning space.
    Instead, we treat this classifier as a proof of the principle that we can
    extend from the linear and quadratic kernels implemented above to higher-
    order polynomials, as demonstrated by the multitude of tasks that prefer
    higher tuned values of `fit__degree`.

    """

    tune_priors = (
        ('fit__degree', (2, 3, 4, 5, 6)),
        ('fit__coef0', (-5, -2, -1, 0, 1, 2, 5)),
        ('fit__C', tuple(10 ** np.linspace(-3.5, 6.5, 41))),
        )
 
    fit_inst = SVC(kernel='poly', gamma=1e-6, probability=True,
                   cache_size=500, class_weight='balanced')


class SVCrbf(Base, Kernel):
    """A support vector classifier using a radial basis kernel.

    The `fit__C` and `fit__gamma` for this classifier exhibit similar
    behaviour to the corresponding hyper-parameters for :class:`SVCquad`
    above, and the values to tune over for both have been chosen using a
    parallel logic.

    """
 
    tune_priors = (
        ('fit__C', tuple(10 ** np.linspace(-1.5, 9.5, 89))),
        ('fit__gamma', tuple(10 ** np.linspace(-7.5, -2.5, 11))),
        )

    fit_inst = SVC(kernel='rbf', probability=True,
                   cache_size=500, class_weight='balanced')


class Forests(Base, Trees):
    """An ensemble classifier using decision trees trained on random subsets.

    Note that tuning over values of `max_features` greater than 10^-0.6
    (i.e. roughly a quarter of the features) would slow down the classifier
    and defeat the purpose of only using a small subset of features at each
    decision step. Likewise, using values of `max_features` smaller than
    10^-3.6 would entail using three or fewer features at each decision step
    in many -omic contexts, making each step increasingly worthless.

    """
 
    tune_priors = (
        ('fit__max_features', tuple(10 ** np.linspace(-3.6, -0.6, 121))),
        ('fit__min_samples_leaf', (1, 2, 3, 4, 6, 8, 10, 15)),
        )

    fit_inst = RandomForestClassifier(n_estimators=1000,
                                      class_weight='balanced')

