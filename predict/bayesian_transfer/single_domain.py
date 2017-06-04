
# Author: Michal Grzadkowski <grzadkow@ohsu.edu>

from ..selection import PathwaySelect

import numpy as np
from scipy import stats

from random import gauss as rnorm
import collections
from sklearn import metrics
from functools import reduce


class BaseSingleDomain(object):

    def __init__(self, prec_alpha, prec_beta, sigma_h):
        self.prec_alpha = prec_alpha
        self.prec_beta = prec_beta
        self.sigma_h = sigma_h

        self.expr_genes = None
        self.path_obj = None
        self.mut_genes = None
        self.X = None
        self.lambda_mat = None
        self.A_mat = None
        self.gamma_list = None
        self.eta_mat = None
        self.bw_mat = None
        self.f_mat = None
        self.H_mat = None

    def log_likelihood(self, kernel_mat, y_list):
        """Computes the log-likelihood of the current model state."""
        if self.lambda_mat is None:
            raise ValueError("Can't compute model likelihood before fitting!")

        # precision prior distribution given precision hyper-parameters
        prec_distr = stats.gamma(a=self.prec_alpha,
                                 scale=self.prec_beta ** -1.0)

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
            stats.norm(loc=self.A_mat['mu'].transpose() @ kernel_mat,
                       scale=self.sigma_h)
                .logpdf(self.H_mat['mu'])
            )

        # likelihood of bias parameter precision priors given
        # precision hyper-parameters
        gamma_logl = np.sum(
            prec_distr.logpdf(np.array(self.gamma_list['alpha'])
                           / np.array(self.gamma_list['beta']))
            )

        # likelihood of bias parameters given their precision priors
        b_logl = np.sum(
            stats.norm(loc=0, scale=(np.array(self.gamma_list['beta'])
                                     / np.array(self.gamma_list['alpha'])))
                .logpdf(self.bw_mat['mu'][:, 0])
            )

        # likelihood of latent feature weight parameter precision priors
        # given precision hyper-parameters
        eta_logl = np.sum(prec_distr.logpdf(self.eta_mat['alpha']
                                            / self.eta_mat['beta']))

        # likelihood of latent feature weight parameters given
        # their precision priors
        w_logl = np.sum(
            stats.norm(loc=0, scale=(self.eta_mat['beta']
                                     / self.eta_mat['alpha']))
                .logpdf(self.bw_mat['mu'][:, 1:])
            )

        # likelihood of predicted outputs given latent features, bias
        # parameters, and latent feature weight parameters
        f_logl = np.sum(
            stats.norm(loc=(self.bw_mat['mu'][:, 1:] @ self.H_mat['mu']
                            + np.vstack(self.bw_mat['mu'][:, 0])),
                       scale=1)
                .logpdf(self.f_mat['mu'])
            )

        # likelihood of actual output labels given class separation margin
        # and predicted output labels
        y_logl = np.sum(
            stats.norm(loc=self.f_mat['mu'] * np.vstack(y_list),
                       scale=self.f_mat['sigma']).logsf(1)
            )

        return (lambda_logl + a_logl + h_logl
                + gamma_logl + b_logl + eta_logl + w_logl + f_logl + y_logl)


class MultiVariant(BaseSingleDomain):

    def __init__(self,
                 kernel, path_keys, latent_features=5,
                 sigma_h=0.1, prec_alpha=1.0, prec_beta=1.0, margin=1.0,
                 max_iter=500, stop_tol=1e-3):
        self.kernel = kernel
        self.path_keys = path_keys
        self.R = latent_features
        self.margin = margin
        self.max_iter = max_iter
        self.stop_tol = stop_tol

        super(MultiVariant, self).__init__(prec_alpha, prec_beta, sigma_h)

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
        """Fits the classifier."""

        # computes the kernel matrices and concatenates them, gets number of
        # training samples and total number of kernel features
        self.X = X
        kernel_mat = self.compute_kernels(X, **fit_params)
        kern_size = kernel_mat.shape[0]
        data_size = kernel_mat.shape[1]

        # makes sure training labels are of the correct format
        if len(y_list) == data_size:
            y_list = np.array(y_list).transpose().tolist()
        y_list = [[1.0 if x else -1.0 for x in y] for y in y_list]

        # initializes matrix of posterior distributions of precision priors
        # for the projection matrix
        lambda_mat = {'alpha': np.array([[self.prec_alpha + 0.5
                                          for _ in range(self.R)]
                                         for _ in range(kern_size)]),
                      'beta': np.array([[self.prec_beta
                                         for _ in range(self.R)]
                                        for _ in range(kern_size)])}

        # initializes posteriors of precision priors for coupled
        # classification matrices
        gamma_list = {'alpha': [self.prec_alpha + 0.5
                                for _ in y_list],
                      'beta': [self.prec_beta for _ in y_list]}
        eta_mat = {'alpha': np.array([[self.prec_alpha + 0.5
                                       for _ in range(self.R)]
                                      for _ in y_list]),
                   'beta': np.array([[self.prec_beta
                                      for _ in range(self.R)]
                                     for _ in y_list])}

        # initializes posterior distributions of projection matrix
        A_mat = {'mu': np.array([[rnorm(0,1)
                                  for _ in range(self.R)]
                                 for _ in range(kern_size)]),
                 'sigma': np.array([np.diag([1.0 for i in range(kern_size)])
                                    for _ in range(self.R)])}

        # initializes posterior distributions of representations
        # in shared sub-space
        H_mat = {'mu': np.array([[rnorm(0,1)
                                  for _ in range(data_size)]
                                 for _ in range(self.R)]),
                 'sigma': np.array(np.diag([1.0 for _ in range(self.R)]))}

        bw_mat = {'mu': np.array([[0] + [rnorm(0,1) for _ in range(self.R)]
                                  for _ in range(len(y_list))]),
                  'sigma': np.array([np.diag([1 for _ in range(self.R + 1)])
                                     for _ in range(len(y_list))])}

        # initializes predicted outputs
        f_mat = {'mu': np.array([[(abs(rnorm(0,1)) + self.margin) * np.sign(i)
                                  for i in y]
                                 for y in y_list]),
                 'sigma': np.array([[1.0 for _ in range(data_size)]
                                    for _ in y_list])}

        # precomputes kernel crossproducts, initializes lower-upper matrix
        kkt_mat = kernel_mat @ kernel_mat.transpose()
        lu_list = [
            {'lower': np.array([-1e40 if i <= 0 else self.margin for i in y]),
             'upper': np.array([1e40 if i >= 0 else -self.margin for i in y])}
            for y in y_list
            ]

        # proceeds with inference using variational Bayes for the given
        # number of iterations
        cur_iter = 1
        old_log_like = float('-inf')
        log_like_stop = False
        while cur_iter <= self.max_iter and not log_like_stop:

            # updates posterior distributions of projection priors
            for j in range(self.R):
                lambda_mat['beta'][:, j] = (
                    self.prec_beta
                    + 0.5 * (np.power(A_mat['mu'][:, j], 2)
                             + np.diag(A_mat['sigma'][j, :, :]))
                )

            # updates posterior distributions of projection matrices
            lambdas_expect = (lambda_mat['alpha']
                              / lambda_mat['beta']).transpose().tolist()
            for j in range(self.R):
                A_mat['sigma'][j, :, :] = np.linalg.inv(
                    np.diag(lambdas_expect[j])
                    + (kkt_mat / self.sigma_h))
                A_mat['mu'][:, j] = np.dot(
                    A_mat['sigma'][j, :, :],
                    np.dot(kernel_mat,
                           H_mat['mu'][j, :].transpose())
                    / self.sigma_h)

            # updates posterior distributions of representations
            H_mat['sigma'] = np.linalg.inv(
                np.diag([self.sigma_h ** -1 for _ in range(self.R)])
                + reduce(lambda x, y: x + y,
                         [np.outer(bw_mat['mu'][j, 1:], bw_mat['mu'][j, 1:])
                          + bw_mat['sigma'][j][1:, 1:]
                          for j in range(len(y_list))])
                )
            H_mat['mu'] = np.dot(
                H_mat['sigma'],
                np.dot(A_mat['mu'].transpose(),
                       kernel_mat) / self.sigma_h
                + reduce(
                    lambda x, y: x + y,
                    [np.outer(bw_mat['mu'][j, 1:], f_mat['mu'][j, ])
                     - np.repeat(a=np.array([
                        [x * bw_mat['mu'][j, 0] + y for x, y in
                         zip(bw_mat['mu'][j, 1:], bw_mat['sigma'][j, 1:, 0])]]
                        ), repeats=data_size, axis=0).transpose()
                     for j in range(len(y_list))]
                    )
                )

            # updates posterior distributions of classification priors
            # in the shared subspace
            gamma_list['beta'] = [self.prec_beta
                                  + 0.5 * (bw_mat['mu'][j, 0] ** 2
                                           + bw_mat['sigma'][j, 0, 0])
                                  for j in range(len(y_list))]

            eta_mat['beta'] = np.vstack([
                self.prec_beta + 0.5 * (
                    np.power(bw_mat['mu'][j, 1:], 2)
                    + np.diag(bw_mat['sigma'][j, 1:, 1:])
                ) for j in range(len(y_list))
                ])

            # updates posterior distributions of classification parameters
            # in the shared subspace
            bw_mat['sigma'] = np.array(
                [np.zeros(shape=(self.R + 1, self.R + 1))
                 for _ in y_list]
                )
            h_sum = np.sum(H_mat['mu'], axis=1)
            hht_mat = (H_mat['mu'] @ H_mat['mu'].transpose()
                       + H_mat['sigma'] * data_size)

            for j in range(len(y_list)):
                bw_mat['sigma'][j, 0, 0] = (
                    gamma_list['alpha'][j] / gamma_list['beta'][j]
                    + data_size
                    )
                bw_mat['sigma'][j, 1:, 0] = h_sum
                bw_mat['sigma'][j, 0, 1:] = h_sum

                bw_mat['sigma'][j, 1:, 1:] = (
                    hht_mat + np.diag(eta_mat['alpha'][j, :]
                                      / eta_mat['beta'][j, :])
                    )

                bw_mat['sigma'][j, :, :] = np.linalg.inv(
                    bw_mat['sigma'][j, :, :])
                bw_mat['mu'][j, :] = np.dot(
                    bw_mat['sigma'][j, :, :],
                    np.dot(np.vstack([np.ones(data_size), H_mat['mu']]),
                           f_mat['mu'][j, :])
                    )

            # updates posterior distributions of predicted outputs
            for j in range(len(y_list)):
                f_out = (H_mat['mu'].transpose() @ bw_mat['mu'][j, 1:]
                         + bw_mat['mu'][j, 0])
                alpha_norm = lu_list[j]['lower'] - f_out
                beta_norm = lu_list[j]['upper'] - f_out
                norm_factor = [stats.norm.cdf(b) - stats.norm.cdf(a) if a != b
                               else 1
                               for a, b in zip(alpha_norm, beta_norm)]

                f_mat['mu'][j, :] = [
                    f + ((stats.norm.pdf(a) - stats.norm.pdf(b)) / n)
                    for a, b, n, f in
                    zip(alpha_norm, beta_norm, norm_factor, f_out)
                    ]
                f_mat['sigma'][j, :] = [
                    1.0 + (a * stats.norm.pdf(a) - b * stats.norm.pdf(b)) / n
                    - ((stats.norm.pdf(a) - stats.norm.pdf(b)) ** 2) / n ** 2
                    for a, b, n in zip(alpha_norm, beta_norm, norm_factor)
                    ]

            self.lambda_mat = lambda_mat
            self.A_mat = A_mat
            self.gamma_list = gamma_list
            self.eta_mat = eta_mat
            self.bw_mat = bw_mat
            self.f_mat = f_mat
            self.H_mat = H_mat

            if (cur_iter % 5) == 0:
                cur_log_like = self.log_likelihood(kernel_mat, y_list)
                if cur_log_like < (old_log_like + 1.0):
                    log_like_stop = True
                else:
                    old_log_like = cur_log_like
                    print('Iteration {}: {}'.format(cur_iter, cur_log_like))

            cur_iter += 1

        return self

    def predict_proba(self, X):
        """Predicts probability of each type of mutation in a new dataset."""

        kernel_mat = self.compute_kernels(x_mat=self.X, y_mat=X)
        data_size = X.shape[0]
        pred_count = self.f_mat['mu'].shape[0]

        h_mu = self.A_mat['mu'].transpose() @ kernel_mat
        f_mu = [[] for _ in range(pred_count)]
        f_sigma = [[] for _ in range(pred_count)]
        pred_list = [[] for _ in range(pred_count)]

        for j in range(pred_count):
            f_mu[j] = (
                np.vstack(([1 for _ in range(data_size)], h_mu)).transpose()
                @ self.bw_mat['mu'][j, :]
                )

            f_sigma[j] = 1.0 + np.diag(
                np.dot(
                    np.dot(np.vstack(
                        ([1 for _ in range(data_size)], h_mu)).transpose(),
                           self.bw_mat['sigma'][j, :, :]),
                    np.vstack(([1 for _ in range(data_size)], h_mu))
                    )
                )

            pred_p = 1 - stats.norm.cdf((self.margin - f_mu[j]) / f_sigma[j])
            pred_n = stats.norm.cdf((-self.margin - f_mu[j]) / f_sigma[j])
            pred_list[j] = pred_p / (pred_p + pred_n)

        return pred_list

    def get_params(self, deep=True):
        return {'sigma_h': self.sigma_h,
                'prec_alpha': self.prec_alpha,
                'prec_beta': self.prec_beta}

    def set_params(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

