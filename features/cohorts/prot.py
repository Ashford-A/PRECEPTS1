
"""Consolidating -omic datasets for prediction of proteomic measurements.

Authors: Michal Grzadkowski <grzadkow@ohsu.edu>

"""

from .base import *

from ..expression import *
from ..proteomics import *

from ..pathways import *
from ..annot import get_gencode
from ..utils import match_tcga_samples

import numpy as np
import pandas as pd


class ProtCohort(ValueCohort, UniCohort):

    def __init__(self,
                 syn, cohort, cv_seed=None, cv_prop=0.8, prot_source=None):

        # gets the prediction features and the abundances to predict
        expr_mat = get_expr_cptac(syn, cohort).fillna(0.0)
        prot_mat = get_prot_cptac(syn, cohort, source=prot_source).fillna(0.0)

        # filters out genes that have both low levels of expression
        # and low variance of expression
        expr_mean = np.mean(expr_mat)
        expr_var = np.var(expr_mat)
        expr_mat = expr_mat.loc[
            :, ((expr_mean > np.percentile(expr_mean, 10))
                | (expr_var > np.percentile(expr_var, 10)))
            ]

        # gets the samples that are common between the datasets, get the
        # training/testing cohort split
        use_samples = set(expr_mat.index) & set(prot_mat.index)
        train_samps, test_samps = self.split_samples(
            cv_seed, cv_prop, use_samples)

        # splits the protein abundances into training/testing sub-cohorts
        self.train_prot = prot_mat.loc[train_samps, :]
        if test_samps:
            self.test_prot = prot_mat.loc[test_samps, :]
        else:
            test_samps = None

        super().__init__(expr_mat, train_samps, test_samps, cohort, cv_seed)

    def train_pheno(self, prot_gene, samps=None):
        if samps is None:
            samps = self.train_samps

        return self.train_prot.loc[samps, prot_gene]

    def test_pheno(self, prot_gene, samps=None):
        if samps is None:
            samps = self.test_samps

        return self.test_prot.loc[samps, prot_gene]

    def train_data(self,
                   pheno,
                   include_samps=None, exclude_samps=None,
                   include_genes=None, exclude_genes=None):

        if isinstance(pheno, str):
            pheno = [pheno]

        return super().train_data(pheno,
                                  include_samps, exclude_samps,
                                  include_genes, exclude_genes)

    def test_data(self,
                  pheno,
                  include_samps=None, exclude_samps=None,
                  include_genes=None, exclude_genes=None):

        if isinstance(pheno, str):
            pheno = [pheno]

        return super().test_data(pheno,
                                 include_samps, exclude_samps,
                                 include_genes, exclude_genes)

