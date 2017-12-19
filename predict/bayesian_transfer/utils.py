
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

