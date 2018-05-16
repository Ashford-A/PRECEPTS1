
import numpy as np


def log_norm(data_mat):
    """Log-normalizes a dataset, usually RNA-seq expression.

    Puts a matrix of continuous values into log-space after adding
    a constant derived from the smallest non-zero value.

    Args:
        data_mat (:obj:`np.array` of :obj:`float`,
                  shape = [n_samples, n_features])

    Returns:
        norm_mat (:obj:`np.array` of :obj:`float`,
                  shape = [n_samples, n_features])

    Examples:
        >>> norm_expr = log_norm(np.array([[1.0, 0], [2.0, 8.0]]))
        >>> print(norm_expr)
                [[ 0.5849625 , -1.],
                 [ 1.32192809,  3.08746284]]

    """
    log_add = np.nanmin(data_mat[data_mat > 0]) * 0.5
    norm_mat = np.log2(data_mat + log_add)

    return norm_mat

