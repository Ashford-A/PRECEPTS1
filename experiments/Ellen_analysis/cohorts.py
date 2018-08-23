
from dryadic.features.cohorts import BaseMutationCohort
from ...features.data.expression import get_expr_toil
from dryadic.features.cohorts.utils import get_gencode, log_norm
from ...features.cohorts.tcga import add_variant_data

import numpy as np
import pandas as pd
import os

from functools import reduce
from operator import and_


class CancerCohort(BaseMutationCohort):
 
    def __init__(self,
                 mut_genes, mut_levels, toil_dir, sample_data, syn, copy_dir,
                 annot_file, top_genes=None, samp_cutoff=25,
                 cv_prop=0.75, cv_seed=None, **coh_args):

        ellen_expr = pd.read_csv(sample_data, sep='\t', index_col=0)
        tcga_expr = get_expr_toil(cohort='BRCA', data_dir=toil_dir,
                                  collapse_txs=True)

        annot_data = get_gencode(annot_file)
        self.gene_annot = {at['gene_name']: {**{'Ens': ens}, **at}
                           for ens, at in annot_data.items()
                           if at['gene_name'] in tcga_expr.columns}

        expr, variants, copy_df = add_variant_data(
            cohort='BRCA', var_source='mc3', copy_source='Firehose', syn=syn,
            expr=tcga_expr, copy_dir=copy_dir, gene_annot=self.gene_annot,
            )

        use_genes = sorted(set(ellen_expr.index) & set(expr.columns))
        expr = expr.loc[:, use_genes]
        ellen_expr = log_norm(ellen_expr.transpose()).loc[:, use_genes]

        ellen_expr = ellen_expr.apply(
            lambda x: (x - x.min()) / (x.max() - x.min()))
        expr = expr.apply(lambda x: (x - x.min()) / (x.max() - x.min()))

        if len(mut_levels) > 1 or mut_levels[0] != 'Gene':
            if 'Gene' in mut_levels:
                scale_lvl = mut_levels.index('Gene') + 1
            else:
                scale_lvl = 0
 
            mut_levels.insert(scale_lvl, 'Scale')
            mut_levels.insert(scale_lvl + 1, 'Copy')
            variants['Scale'] = 'Point'
            copy_df['Scale'] = 'Copy'

        super().__init__(pd.concat([expr, ellen_expr], sort=True).fillna(0.0),
                         pd.concat([variants, copy_df], sort=True),
                         mut_genes, mut_levels, top_genes, samp_cutoff,
                         cv_prop, cv_seed)

