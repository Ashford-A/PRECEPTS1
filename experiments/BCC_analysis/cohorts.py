
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
                 mut_genes, mut_levels, toil_dir, patient_dir, syn, copy_dir,
                 annot_file, top_genes=None, samp_cutoff=25,
                 cv_prop=0.75, cv_seed=None, **coh_args):

        bcc_expr = pd.read_csv(os.path.join(patient_dir, "tatlow_kallisto",
                                            "BCC_gene-level_tpm.tsv"),
                               sep='\t', index_col=0)

        tcga_expr = get_expr_toil(cohort='PAAD', data_dir=toil_dir,
                                  collapse_txs=True)

        annot_data = get_gencode(annot_file)
        self.gene_annot = {at['gene_name']: {**{'Ens': ens}, **at}
                           for ens, at in annot_data.items()
                           if at['gene_name'] in tcga_expr.columns}

        expr, variants, copy_df = add_variant_data(
            cohort='PAAD', var_source='mc3', copy_source='Firehose', syn=syn,
            expr=tcga_expr, copy_dir=copy_dir, gene_annot=self.gene_annot,
            )

        gene_ids = [self.gene_annot[gn]['gene_id'].split('.')[0]
                    for gn in expr.columns]
        bcc_expr.index = [gn.split('.')[0] for gn in bcc_expr.index]
        bcc_expr = bcc_expr.loc[bcc_expr.index.isin(gene_ids), :]
        bcc_expr = log_norm(bcc_expr.loc[gene_ids, :].transpose())
        bcc_expr.columns = expr.columns

        if len(mut_levels) > 1 or mut_levels[0] != 'Gene':
            if 'Gene' in mut_levels:
                scale_lvl = mut_levels.index('Gene') + 1
            else:
                scale_lvl = 0
 
            mut_levels.insert(scale_lvl, 'Scale')
            mut_levels.insert(scale_lvl + 1, 'Copy')
            variants['Scale'] = 'Point'
            copy_df['Scale'] = 'Copy'

        super().__init__(pd.concat([expr, bcc_expr], sort=True),
                         pd.concat([variants, copy_df], sort=True),
                         mut_genes, mut_levels, top_genes, samp_cutoff,
                         cv_prop, cv_seed)

