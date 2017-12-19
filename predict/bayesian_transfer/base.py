
class BaseBayesianTransfer(object):

    def __init__(self,
                 path_keys=None, kernel='rbf', latent_features=5,
                 prec_distr=(1.0, 1.0), sigma_h=0.1, margin=1.0,
                 kern_gamma=-1.85, max_iter=200, stop_tol=1.0):
        self.kernel = kernel
        self.path_keys = path_keys
        self.R = latent_features

        self.prec_distr = prec_distr
        self.sigma_h = sigma_h
        self.margin = margin
        self.kern_gamma = kern_gamma

        self.max_iter = max_iter
        self.stop_tol = stop_tol

        self.expr_genes = None
        self.path_obj = None
        self.mut_genes = None

        self.weight_priors = None
        self.task_count = None

    def get_params(self, deep=True):
        return {'sigma_h': self.sigma_h,
                'prec_distr': self.prec_distr,
                'latent_features': self.R,
                'margin': self.margin,
                'kern_gamma': self.kern_gamma}

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
                    gamma=(np.mean(metrics.pairwise.pairwise_distances(x))
                           ** self.kern_gamma)
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

    def transform(self, X):
        return X

    @abstractmethod
    def get_pred_class_probs(self, pred_mu, pred_sigma):
        """Gets the predicted probability of falling into output classes."""

    @abstractmethod
    def init_output_mat(self, y_list):
        """Initialize posterior distributions of the output predictions."""

    def update_precision_priors(self,
                                precision_mat, variable_mat,
                                prec_alpha, prec_beta):
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
                                + prec_alpha + 0.5),
                      'beta': (prec_beta
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

