
"""Using transfer learning to predict one or more labels in a single domain.

This module contains classes corresponding to particular versions of Bayesian
transfer learning algorithms for predicting binary labels in one or more
tasks. Learning is done within one domain using one or more kernels to map the
decision boundary to higher dimensions using pathway-based feature selection.
Optimizing over the set of unknown parameters and latent variables in each
model is done using variational inference, in which the posterior probability
of each is approximated analytically.

Authors: Michal Grzadkowski <grzadkow@ohsu.edu>
         Hannah Manning <manningh@ohsu.edu>

"""

from ..selection import PathwaySelect

import numpy as np
from scipy import stats
from abc import abstractmethod

import collections
from sklearn import metrics
from functools import reduce


def get_square_gauss(x_mat):
    """Calculates the expected squared value of a set of gaussian variables.

    Args:
        x_mat (dict): A matrix of gaussian variables defined by a 'mu'
                      element and a 'sigma' element.

    Returns:
        sq_mat (numpy array of float)

    Examples:
        >>> test_mat = {'mu': np.array([[0., 1., 0.],
        >>>                             [-1., 0., 2.]]),
        >>>             'sigma': np.array([[[1., 1.], [0.5, 1.]],
        >>>                                [[0.5, 1.], [1., 0.]],
        >>>                                [[2., 0.], [0., 1.]]])}
        >>> print(get_square_gauss(test_mat))
                [[1., 1.5, 2.],
                 [2., 0., 5.]]

    """
    sq_mat = np.zeros(x_mat['mu'].shape)

    for i in range(x_mat['mu'].shape[1]):
        sq_mat[:, i] = x_mat['mu'][:, i] ** 2.0
        sq_mat[:, i] += np.diag(x_mat['sigma'][i, :, :])

    return sq_mat


class BaseSingleDomain(object):
    """Base class for transfer learning classifiers in one domain.

    Args:

    Attributes:

    """

    def __init__(self,
                 path_keys=None, kernel='rbf', latent_features=5,
                 prec_distr=(1.0, 1.0), sigma_h=0.1, margin=1.0,
                 max_iter=100, stop_tol=1.0):
        self.kernel = kernel
        self.path_keys = path_keys
        self.R = latent_features

        self.prec_distr = prec_distr
        self.sigma_h = sigma_h
        self.margin = margin
        self.max_iter = max_iter
        self.stop_tol = stop_tol

        self.expr_genes = None
        self.path_obj = None
        self.mut_genes = None

        self.lambda_mat = None
        self.A_mat = None
        self.H_mat = None

        self.weight_priors = None
        self.weight_mat = None
        self.output_mat = None

        self.X = None
        self.kernel_mat = None
        self.kkt_mat = None

        self.kern_size = None
        self.sample_count = None
        self.task_count = None

    def get_params(self, deep=True):
        return {'sigma_h': self.sigma_h,
                'prec_distr': self.prec_distr,
                'latent_features': self.R}

    def set_params(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

    def compute_kernels(self, x_mat, y_mat=None, **fit_params):
        """Gets the kernel matrices from a list of feature matrices."""

        for param in ('path_keys', 'path_obj', 'mut_genes'):
            if self.__getattribute__(param) is None:
                self.__setattr__(param, fit_params[param])
            else:
                fit_params = {**fit_params,
                              **{param: self.__getattribute__(param)}}
        if self.expr_genes is None:
            self.expr_genes = fit_params['expr_genes']

        select_list = [PathwaySelect(pk, expr_genes=self.expr_genes)
                       for pk in self.path_keys]
        x_list = [ps.fit(X=x_mat, y=None, **fit_params).transform(x_mat)
                  for ps in select_list]

        if y_mat is not None:
            y_list = [ps.fit(X=y_mat, y=None, **fit_params).transform(y_mat)
                      for ps in select_list]
        else:
            y_list = x_list

        if isinstance(self.kernel, collections.Callable):
            kernel_list = [self.kernel(x, y) for x, y in zip(x_list, y_list)]

        elif self.kernel == 'rbf':
            kernel_list = [
                metrics.pairwise.rbf_kernel(
                    x, y,
                    gamma=np.mean(
                        metrics.pairwise.pairwise_distances(x)) ** -2.0
                    )
                for x, y in zip(x_list, y_list)
                ]

        elif self.kernel == 'linear':
            kernel_list = [metrics.pairwise.linear_kernel(x, y)
                           for x, y in zip(x_list, y_list)]

        else:
            raise ValueError(
                "Unknown kernel " + str(self.kernel) + " specified!")

        return np.vstack(kernel_list)

    def fit(self, X, y_list, verbose=False, **fit_params):
        """Fits the classifier.

        Args:
            X (array-like of float), shape = [n_samples, n_features]
            y_list (array-like of bool): Known output labels.
            verbose (bool): How much information to print during fitting.
            fit_params (dict): Other parameters to control fitting process.

        Returns:
            self (MultiVariant): The fitted instance of the classifier.

        """
        self.X = X

        # computes the kernel matrices and concatenates them, gets number of
        # training samples and total number of kernel features
        self.kernel_mat = self.compute_kernels(X, **fit_params)
        self.kkt_mat = self.kernel_mat @ self.kernel_mat.transpose()
        self.kern_size = self.kernel_mat.shape[0]
        self.sample_count = self.kernel_mat.shape[1]

        # makes sure training labels are of the correct format
        if len(y_list) == self.sample_count:
            y_list = np.array(y_list).transpose().tolist()
        y_list = [[1.0 if x else -1.0 for x in y] for y in y_list]
        self.task_count = len(y_list)

        # initializes matrix of posterior distributions of precision priors
        # for the projection matrix
        self.lambda_mat = {'alpha': (np.zeros((self.kern_size, self.R))
                                     + self.prec_distr[0] + 0.5),
                           'beta': (np.zeros((self.kern_size, self.R))
                                    + self.prec_distr[1])}

        # initializes posteriors of precision priors for coupled
        # classification matrices
        self.weight_priors = {
            'alpha': (np.zeros((self.R + 1, self.task_count))
                      + self.prec_distr[0] + 0.5),
            'beta': np.zeros((self.R + 1, self.task_count)) + self.prec_distr[1]
            }

        self.A_mat = {'mu': np.random.randn(self.kern_size, self.R),
                      'sigma': np.array(np.eye(self.kern_size)[..., None]
                                        * ([1] * self.R)).transpose()}

        self.H_mat = {'mu': np.random.randn(self.R, self.sample_count),
                      'sigma': np.eye(self.R)}

        self.weight_mat = {
            'mu': np.vstack((np.zeros((1, self.task_count)),
                             np.random.randn(self.R, self.task_count))),
            'sigma': np.tile(np.eye(self.R + 1), (self.task_count, 1, 1))
            }

        self.output_mat = self.init_output_mat(y_list)
        lu_list = self.get_lu_list(y_list)

        # proceeds with inference using variational Bayes for the given
        # number of iterations
        cur_iter = 1
        old_log_like = float('-inf')
        log_like_stop = False
        while cur_iter <= self.max_iter and not log_like_stop:

            new_lambda = self.update_precision_priors(self.lambda_mat,
                                                      self.A_mat)
            new_proj = self.update_projection(new_lambda,
                                              self.A_mat, self.H_mat)
            new_latent = self.update_latent(new_proj,
                                            self.weight_mat, self.output_mat)

            new_weight_priors = self.update_precision_priors(
                self.weight_priors, self.weight_mat)
            new_weights = self.update_weights(
                new_weight_priors, new_latent, self.output_mat)
            new_outputs = self.update_output(new_latent, new_weights,
                                             y_list, lu_list)

            self.lambda_mat = new_lambda
            self.A_mat = new_proj
            self.H_mat = new_latent

            self.weight_priors = new_weight_priors
            self.weight_mat = new_weights
            self.output_mat = new_outputs

            if (cur_iter % 5) == 0:
                cur_log_like = self.log_likelihood(y_list)

                if cur_log_like < (old_log_like + self.stop_tol):
                    log_like_stop = True

                else:
                    old_log_like = cur_log_like

                    if verbose:
                        print('Iteration {}: {}'.format(
                            cur_iter, cur_log_like))

            cur_iter += 1

        return self

    def predict_proba(self, X):
        kern_dist = self.compute_kernels(x_mat=self.X, y_mat=X)
        predict_samps = X.shape[0]

        h_mu = self.A_mat['mu'].transpose() @ kern_dist
        f_mu = [np.zeros(predict_samps) for _ in range(self.task_count)]
        f_sigma = [np.zeros(predict_samps) for _ in range(self.task_count)]
        pred_list = [np.zeros(predict_samps) for _ in range(self.task_count)]

        for i in range(self.task_count):
            f_mu[i] = np.dot(
                np.vstack(([1 for _ in range(predict_samps)], h_mu))
                    .transpose(),
                self.weight_mat['mu'][:, i]
                )

            f_sigma[i] = 1.0 + np.diag(
                np.dot(
                    np.dot(
                        np.vstack(([1 for _ in
                                    range(predict_samps)], h_mu)).transpose(),
                        self.weight_mat['sigma'][i, :, :]),
                    np.vstack(([1 for _ in range(predict_samps)], h_mu))
                    )
                )

            pred_p = 1 - stats.norm.cdf((self.margin - f_mu[i]) / f_sigma[i])
            pred_n = stats.norm.cdf((-self.margin - f_mu[i]) / f_sigma[i])
            pred_list[i] = pred_p / (pred_p + pred_n)

        return pred_list

    @abstractmethod
    def init_output_mat(self, y_list):
        """Initialize posterior distributions of the output predictions."""

    @abstractmethod
    def get_lu_list(self, y_list):
        """Gets the truncations for the posterior output distributions."""

    def update_precision_priors(self, precision_mat, variable_mat):
        """Updates the posterior distributions of a set of precision priors.

        Performs an update step for the approximate posterior distributions
        of a matrix of gamma-distributed precision priors for a set of
        normally-distributed downstream variables.

        Args:
            precision_mat (dict): Current precision prior posteriors.
            variable_mat (dict): Current downstream variable posteriors.

        Returns:
            new_priors (dict): Updated precision priors.

        """
        new_priors = {'alpha': (np.zeros(precision_mat['alpha'].shape)
                                + self.prec_distr[0] + 0.5),
                      'beta': (self.prec_distr[1]
                               + 0.5 * get_square_gauss(variable_mat))}

        return new_priors

    def update_projection(self, prior_mat, variable_mat, feature_mat):
        """Updates posterior distributions of projection matrices.

        Args:

        Returns:

        """
        new_variable = {'mu': np.zeros(variable_mat['mu'].shape),
                        'sigma': np.zeros(variable_mat['sigma'].shape)}
        prior_expect = (prior_mat['alpha'] / prior_mat['beta'])\
            .transpose().tolist()

        for i in range(self.R):
            new_variable['sigma'][i, :, :] = np.linalg.inv(
                np.diag(prior_expect[i])
                + (self.kkt_mat / self.sigma_h))
            new_variable['mu'][:, i] = np.dot(
                new_variable['sigma'][i, :, :],
                np.dot(self.kernel_mat,
                       feature_mat['mu'][i, :].transpose())
                / self.sigma_h)

        return new_variable

    def update_latent(self, variable_mat, weight_mat, output_mat):
        """Updates latent feature matrix.

        Args:

        Returns:

        """
        new_latent = {k: np.zeros(self.H_mat[k].shape) for k in self.H_mat}

        new_latent['sigma'] = np.linalg.inv(
            np.diag([self.sigma_h ** -1 for _ in range(self.R)])
            + reduce(lambda x, y: x + y,
                     [np.outer(weight_mat['mu'][1:, i],
                               weight_mat['mu'][1:, i])
                      + weight_mat['sigma'][i][1:, 1:]
                      for i in range(self.task_count)])
            )

        new_latent['mu'] = np.dot(
            new_latent['sigma'],
            np.dot(variable_mat['mu'].transpose(),
                   self.kernel_mat) / self.sigma_h
            + reduce(
                lambda x, y: x + y,
                [np.outer(weight_mat['mu'][1:, i], output_mat['mu'][i, :])
                 - np.repeat(a=np.array([
                    [x * weight_mat['mu'][0, i] + y for x, y in
                     zip(weight_mat['mu'][1:, i],
                         weight_mat['sigma'][i, 1:, 0])]]
                    ), repeats=self.sample_count, axis=0).transpose()
                 for i in range(self.task_count)]
                )
            )

        return new_latent

    def update_weights(self, weight_priors, latent_mat, output_mat):
        """Update the weights.

        """
        new_weights = {'mu': np.zeros(weight_priors['alpha'].shape),
                       'sigma': np.zeros((self.task_count,
                                          self.R + 1, self.R + 1))}

        h_sum = np.sum(latent_mat['mu'], axis=1)
        hht_mat = (latent_mat['mu'] @ latent_mat['mu'].transpose()
                   + latent_mat['sigma'] * self.sample_count)

        for i in range(self.task_count):
            new_weights['sigma'][i, 0, 0] = (
                weight_priors['alpha'][0, i] / weight_priors['beta'][0, i]
                + self.sample_count
                )
            new_weights['sigma'][i, 1:, 0] = h_sum
            new_weights['sigma'][i, 0, 1:] = h_sum

            new_weights['sigma'][i, 1:, 1:] = (
                hht_mat + np.diag(weight_priors['alpha'][1:, i]
                                  / weight_priors['beta'][1:, i])
                )

            new_weights['sigma'][i, :, :] = np.linalg.inv(
                new_weights['sigma'][i, :, :])
            new_weights['mu'][:, i] = np.dot(
                new_weights['sigma'][i, :, :],
                np.dot(np.vstack([np.ones(self.sample_count),
                                  latent_mat['mu']]),
                       output_mat['mu'][i, :])
                )

        return new_weights

    @abstractmethod
    def update_output(self, latent_mat, weight_mat, y_list, lu_list):
        """Update the predicted output labels."""

    @abstractmethod
    def get_y_logl(self, y_list):
        """Computes the log-likelihood of a set of output labels given
           the current model state.
        """

    def log_likelihood(self, y_list):
        """Computes the log-likelihood of the current model state.

        Args:

        Returns:

        """
        if self.lambda_mat is None:
            raise ValueError("Can't compute model likelihood before fitting!")

        # precision prior distribution given precision hyper-parameters
        prec_distr = stats.gamma(a=self.prec_distr[0],
                                 scale=self.prec_distr[1] ** -1.0)

        # likelihood of projection matrix precision priors given
        # precision hyper-parameters
        lambda_logl = np.sum(
            prec_distr.logpdf(self.lambda_mat['alpha']
                              / self.lambda_mat['beta'])
            )

        # likelihood of projection matrix values given their precision priors
        a_logl = np.sum(
            stats.norm(loc=0, scale=(self.lambda_mat['beta']
                                     / self.lambda_mat['alpha']))
                .logpdf(self.A_mat['mu'])
            )

        # likelihood of latent feature matrix given kernel matrix,
        # projection matrix, and standard deviation hyper-parameter
        h_logl = np.sum(
            stats.norm(loc=self.A_mat['mu'].transpose() @ self.kernel_mat,
                       scale=self.sigma_h)
                .logpdf(self.H_mat['mu'])
            )

        # likelihood of bias parameter precision priors given
        # precision hyper-parameters
        weight_prior_logl = np.sum(
            prec_distr.logpdf(np.array(self.weight_priors['alpha'])
                              / np.array(self.weight_priors['beta']))
            )

        # likelihood of bias parameters given their precision priors
        weight_logl = np.sum(
            stats.norm(loc=0, scale=(np.array(self.weight_priors['beta'])
                                     / np.array(self.weight_priors['alpha'])))
                .logpdf(self.weight_mat['mu'])
            )

        # likelihood of predicted outputs given latent features, bias
        # parameters, and latent feature weight parameters
        f_logl = np.sum(
            stats.norm(
                loc=(self.weight_mat['mu'][1:, :].transpose()
                     @ self.H_mat['mu']
                     + np.vstack(self.weight_mat['mu'][0, :])),
                scale=1).logpdf(self.output_mat['mu'])
            )

        # likelihood of actual output labels given class separation margin
        # and predicted output labels
        y_logl = np.sum(self.get_y_logl(y_list))

        return (lambda_logl + a_logl + h_logl
                + weight_prior_logl + weight_logl + f_logl + y_logl)

    def get_output_distr(self, stat_list):
        """Gets the cumulative PDF of the output labels.

        Returns:
            out_probs (list): For each task, a list of (x, prob) pairs
        """
        out_probs = [[np.zeros(500) for _ in stat_list]
                     for _ in range(self.task_count)]

        # for each task, get the posterior distribution for each predicted
        # output label
        for i in range(self.task_count):
            distr_vec = [
                stats.norm(loc=mu, scale=sigma) for mu, sigma in
                zip(self.output_mat['mu'][i, :],
                    self.output_mat['sigma'][i, :])
                ]

            # calculate the range of possible predicted output values
            min_range = min(distr.ppf(0.01) for distr in distr_vec)
            max_range = max(distr.ppf(0.99) for distr in distr_vec)
            x_vals = np.linspace(min_range, max_range, 1000)

            # calculate the cumulative probability density function across all
            # label distributions at each possible predicted value
            for j, stat in enumerate(stat_list):
                out_probs[i][j] = [
                    (x,
                     np.mean([distr.pdf(x) for distr, st in
                              zip(distr_vec, stat) if st])
                     ) for x in x_vals
                    ]

        return out_probs

    def get_path_prior(self):
        proj_max = np.apply_along_axis(lambda x: max(abs(x)), 1,
                                       self.A_mat['mu'])
        proj_list = {}

        for i, pk in enumerate(self.path_keys):
            start_indx = self.sample_count * i
            end_indx = self.sample_count * (i + 1)
            proj_list[pk] = proj_max[start_indx:end_indx]

        return proj_list


class MultiVariant(BaseSingleDomain):

    def init_output_mat(self, y_list):
        """Initialize posterior distributions of the output predictions."""
        output_mat = {
            'mu': (abs(np.random.randn(self.task_count, self.sample_count))
                   + self.margin),
            'sigma': np.ones((self.task_count, self.sample_count))
            }
        for i in range(self.task_count):
            output_mat['mu'][i, :] *= np.sign(y_list[i])

        return output_mat

    def get_lu_list(self, y_list):
        return [{'lower': np.array([-1e40 if i <= 0 else self.margin
                                    for i in y]),
                 'upper': np.array([1e40 if i >= 0 else -self.margin
                                    for i in y])}
                for y in y_list]

    def update_output(self, latent_mat, weight_mat, y_list, lu_list):
        """Update the predicted output labels."""
        new_output = {k: np.zeros(self.output_mat[k].shape)
                      for k in self.output_mat}

        for i in range(self.task_count):
            f_raw = np.dot(np.tile(weight_mat['mu'][1:, i], (1, 1)),
                           latent_mat['mu']) + weight_mat['mu'][0, i]

            alpha_norm = (lu_list[i]['lower'] - f_raw)[0, :].tolist()
            beta_norm = (lu_list[i]['upper'] - f_raw)[0, :].tolist()
            norm_factor = [stats.norm.cdf(b) - stats.norm.cdf(a) if a != b
                           else 1
                           for a, b in zip(alpha_norm, beta_norm)]

            new_output['mu'][i, :] = [
                f + ((stats.norm.pdf(a) - stats.norm.pdf(b)) / n)
                for a, b, n, f in
                zip(alpha_norm, beta_norm, norm_factor, f_raw[0, :].tolist())
                ]
            new_output['sigma'][i, :] = [
                1.0 + (a * stats.norm.pdf(a) - b * stats.norm.pdf(b)) / n
                - ((stats.norm.pdf(a) - stats.norm.pdf(b)) ** 2) / n ** 2
                for a, b, n in zip(alpha_norm, beta_norm, norm_factor)
                ]

        return new_output

    def get_y_logl(self, y_list):
        return stats.norm(loc=self.output_mat['mu'] * np.vstack(y_list),
                          scale=self.output_mat['sigma']).logsf(1)


class MultiVariantAsym(BaseSingleDomain):
    """A multi-task transfer learning classifier with assymetric margins.

    Args:

    Attributes:

    """

    def __init__(self,
                 path_keys=None, kernel='rbf', latent_features=5,
                 prec_distr=(1.0, 1.0), sigma_h=0.1, margin=1.0,
                 max_iter=100, stop_tol=1.0):
        super(MultiVariantAsym, self).__init__(
            path_keys=path_keys, kernel=kernel,
            latent_features=latent_features, prec_distr=prec_distr,
            sigma_h=sigma_h, margin=5.0/3,
            max_iter=max_iter, stop_tol=stop_tol)

    def init_output_mat(self, y_list):
        """Initialize posterior distributions of the output predictions."""
        output_mat = {
            'mu': np.array([[stats.norm.rvs(self.margin, self.margin ** -1.0)
                             if y == 1.0
                             else stats.norm.rvs(0.0, 1.0)
                             for y in y_vec] for y_vec in y_list]),
            'sigma': np.array([[self.margin ** -1.0 if y == 1.0 else 1.0
                                for y in y_vec] for y_vec in y_list])
            }

        return output_mat

    def get_lu_list(self, y_list):
        return [{'lower': np.array([-1e40 if i <= 0 else 0.0
                                    for i in y]),
                 'upper': np.array([1e40 if i >= 0 else self.margin
                                    for i in y])}
                for y in y_list]

    def update_output(self, latent_mat, weight_mat, y_list, lu_list):
        """Update the predicted output labels."""
        new_output = {k: np.zeros(self.output_mat[k].shape)
                      for k in self.output_mat}

        for i in range(self.task_count):
            f_raw = np.dot(np.tile(weight_mat['mu'][1:, i], (1, 1)),
                           latent_mat['mu']) + weight_mat['mu'][0, i]

            alpha_norm = (lu_list[i]['lower'] - f_raw)[0, :].tolist()
            beta_norm = (lu_list[i]['upper'] - f_raw)[0, :].tolist()
            pos_distr = stats.norm(loc=self.margin, scale=self.margin ** -1.0)
            neg_distr = stats.norm(loc=0.0, scale=1)

            norm_factor = [
                1.0 if a == b
                else pos_distr.cdf(b) - pos_distr.cdf(a) if y == 1.0
                else neg_distr.cdf(b) - neg_distr.cdf(a)
                for a, b, y in zip(alpha_norm, beta_norm, y_list[i])
                ]

            new_output['mu'][i, :] = [
                f + ((pos_distr.pdf(a) - pos_distr.pdf(b)) / n) if y == 1.0
                else f + ((neg_distr.pdf(a) - neg_distr.pdf(b)) / n)
                for a, b, n, f, y in
                zip(alpha_norm, beta_norm, norm_factor,
                    f_raw[0, :].tolist(), y_list[i])
                ]

            new_output['sigma'][i, :] = [
                self.margin ** -1.0 + (
                    (a * pos_distr.pdf(a) - b * pos_distr.pdf(b)) / n
                    - ((pos_distr.pdf(a) - pos_distr.pdf(b)) ** 2) / n ** 2
                    ) if y == 1.0 else
                1.0 + (
                    (a * neg_distr.pdf(a) - b * neg_distr.pdf(b)) / n
                    - ((neg_distr.pdf(a) - neg_distr.pdf(b)) ** 2) / n ** 2
                )
                for a, b, n, y in zip(alpha_norm, beta_norm, norm_factor,
                                      y_list[i])
                ]

        return new_output

    def get_y_logl(self, y_list):
        y_hat = [[0.0 if i == -1.0 else self.margin for i in y]
                 for y in y_list]

        distr_mat = stats.norm(loc=self.output_mat['mu'] * np.vstack(y_hat),
                               scale=self.output_mat['sigma'])
        cutoff_mat = np.array([[0.0 if i == 1.0 else -self.margin for i in y]
                               for y in y_list])
        y_logl = np.sum(distr_mat.sf(cutoff_mat))

        return y_logl

    def get_params(self, deep=True):
        return {'sigma_h': self.sigma_h,
                'prec_distr': self.prec_distr,
                'latent_features': self.R,
                'margin': self.margin}
