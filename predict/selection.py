
"""
HetMan (Heterogeneity Manifold)
Prediction of mutation sub-types using expression data.
This file contains the methods used to select genetic features for use
in downstream prediction algorithms.
"""

# Author: Michal Grzadkowski <grzadkow@ohsu.edu>

from itertools import chain
import numpy as np
from sklearn.feature_selection.base import SelectorMixin


class PathwaySelect(SelectorMixin):
    """Chooses gene features based on their presence
       in Pathway Commons pathways.
    """

    def __init__(self, path_keys=None, expr_genes=None):
        if isinstance(path_keys, set) or path_keys is None:
            self.path_keys = path_keys
        else:
            self.path_keys = {path_keys}

        self.select_genes = None
        self.expr_genes = expr_genes
        super(PathwaySelect, self).__init__()

    def fit(self, X, y=None, **fit_params):
        """Gets the list of genes selected based on pathway information."""

        fit_params = {k.split('__')[-1]: v for k, v in fit_params.items()}

        if self.path_keys is None:
            select_genes = set(X.columns)

        else:
            path_obj = fit_params['path_obj']
            select_genes = set()

            for gene in fit_params['mut_genes']:
                for path_key in self.path_keys:
                    for pdirs, ptypes in path_key:

                        if len(pdirs) == 0:
                            select_genes |= set(chain(*chain(
                                *[[g for t, g in v.items() if t in ptypes]
                                  for v in path_obj[gene].values()]
                                )))

                        elif len(ptypes) == 0:
                            select_genes |= set(chain(*chain(
                                *[v.values()
                                  for k, v in path_obj[gene].items()
                                  if k in pdirs]
                                )))

                        else:
                            select_genes |= set(chain(*chain(
                                *[[g for t, g in v.items() if t in ptypes]
                                  for k, v in path_obj[gene].items()
                                  if k in pdirs]
                                )))

        self.select_genes = select_genes - set(fit_params['mut_genes'])
        if hasattr(X, 'columns'):
            self.expr_genes = X.columns

        return self

    def _get_support_mask(self):
        """Gets the index of selected genes used to subset a matrix."""
        if self.select_genes is None:
            raise ValueError("PathwaySelect instance has not been fit yet!")
        else:
            return np.array([g in self.select_genes for g in self.expr_genes])
    
    def set_params(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

    def get_params(self, deep=True):
        return {'path_keys': self.path_keys}

