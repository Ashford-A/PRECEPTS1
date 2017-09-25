
"""Selection of genetic features for machine learning using -omics datasets.

# Author: Michal Grzadkowski <grzadkow@ohsu.edu>

"""

from itertools import chain
from functools import reduce

import numpy as np
from sklearn.base import TransformerMixin


class PathwaySelect(TransformerMixin):
    """Chooses gene features based on their presence
       in Pathway Commons pathways.
    """

    def __init__(self, path_keys=None):
        if isinstance(path_keys, set) or path_keys is None:
            self.path_keys = path_keys
        else:
            self.path_keys = {path_keys}

        self.select_genes = None
        self.expr_genes = None

        super().__init__()

    def fit(self, X, y=None, path_obj=None, expr_genes=None, **fit_params):
        self.expr_genes = expr_genes

        if self.path_keys is None:
            if hasattr(X, 'shape'):
                select_genes = set([xcol.split('__')[-1]
                                    for xcol in X.columns])

            else:
                select_genes = [
                    set([xcol.split('__')[-1] for xcol in Xmat.columns])
                    for Xmat in X
                    ]

            self.select_genes = select_genes

        return self

    def transform(self, X):
        mask = self._get_support_mask()

        if isinstance(mask, dict):
            return {lbl: X[lbl].loc[:, mask[lbl]] for lbl in mask}
        else:
            return np.array(X)[:, mask]

    def _get_support_mask(self):
        """Gets the index of selected genes used to subset a matrix."""
        if self.select_genes is None:
            raise ValueError("PathwaySelect instance has not been fit yet!")

        if isinstance(self.expr_genes, dict):
            return {lbl: [g in self.select_genes
                          for g in self.expr_genes[lbl]]
                    for lbl in self.expr_genes}

        else:
            return np.array([g in self.select_genes for g in self.expr_genes])

    def get_params(self, deep=True):
        return {'path_keys': self.path_keys}
    
    def set_params(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)


class NeighbourSelect(PathwaySelect):

    def fit(self, X, y=None, path_obj=None, genes=None, expr_genes=None, **fit_params):
        """Gets the list of genes selected based on pathway information.

        """
        if self.path_keys is not None:
            select_genes = set()

            for gene in genes:
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

            self.select_genes = select_genes - set(genes)

        return super().fit(X, y, path_obj, genes, fit_params)


class IntxTypeSelect(PathwaySelect):

    def fit(self, X, y=None, path_obj=None, expr_genes=None, prot_genes=None,
            **fit_params):
        """Gets the list of genes selected based on pathway information.

        """
        if self.path_keys is not None:
            select_genes = set()

            for intx_type in self.path_keys:
                select_genes |= reduce(lambda x, y: set(x) | set(y),
                                       path_obj[intx_type])

            self.select_genes = select_genes | set(prot_genes)

        return super().fit(X, y, path_obj, expr_genes)

