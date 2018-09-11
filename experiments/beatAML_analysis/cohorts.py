
from dryadic.features.cohorts import BaseMutationCohort
from dryadic.features.cohorts.utils import get_gencode, log_norm
from ...features.cohorts.tcga import add_variant_data

import numpy as np
import pandas as pd
import os

from functools import reduce
from operator import itemgetter, and_


def get_expr_toil(data_dir, tx_map, cohort='LAML', collapse_txs=False):
    expr = pd.read_csv(os.path.join(data_dir, 'TCGA',
                                    'TCGA_{}_tpm.tsv.gz'.format(cohort)),
                       sep='\t')

    expr.index = expr.iloc[:, 0].str.split('|').apply(itemgetter(0))
    expr = expr.iloc[:, 1:]
    expr.index.name = 'Transcript'

    id_map = pd.read_csv(os.path.join(data_dir, 'TCGA_ID_MAP.csv'),
                         sep=',', index_col=0)
    id_map = id_map.loc[id_map['Disease'] == cohort]
    expr.columns = id_map.loc[expr.columns, 'AliquotBarcode'].values

    tx_annot = pd.read_csv(tx_map, sep='\t', index_col=0)
    expr = expr.transpose().loc[:, [tx in tx_annot.index
                                    for tx in expr.index]]

    expr.columns = pd.MultiIndex.from_arrays(
        [tx_annot.loc[expr.columns, 'gene'], expr.columns],
        names=['Gene', 'Transcript']
        )
    
    if collapse_txs:
        expr = np.log2(expr.rpow(2).subtract(0.001).groupby(
            level=['Gene'], axis=1).sum().add(0.001))

    else:
        expr.sort_index(axis=1, level=['Gene'], inplace=True)

    return expr


class CancerCohort(BaseMutationCohort):
 
    def __init__(self,
                 mut_genes, mut_levels, toil_dir, sample_data, tx_map,
                 syn, copy_dir, annot_file, top_genes=None, samp_cutoff=25,
                 cv_prop=0.75, cv_seed=None, **coh_args):
        beataml_expr = pd.read_csv(
            os.path.join(sample_data, 'matrices', 'CTD2_TPM_transcript.tsv'),
            sep='\t', index_col=0
            )

        tcga_expr = get_expr_toil(data_dir=toil_dir, tx_map=tx_map)
        annot_data = get_gencode(annot_file)
        self.gene_annot = {at['gene_name']: {**{'Ens': ens}, **at}
                           for ens, at in annot_data.items()
                           if at['gene_name'] in tcga_expr.columns}

        expr, variants, copy_df = add_variant_data(
            cohort='LAML', var_source='mc3', copy_source='Firehose', syn=syn,
            expr=tcga_expr, copy_dir=copy_dir, gene_annot=self.gene_annot,
            )

        use_genes = sorted(set(beataml_expr.index)
                           & set(expr.columns.get_level_values('Transcript')))
        expr = log_norm(expr.loc[:, (slice(None), use_genes)])
        beataml_expr = log_norm(beataml_expr.transpose()).loc[:, use_genes]
        expr.columns = expr.columns.get_level_values('Transcript')

        beataml_expr = beataml_expr.fillna(0.0).apply(
            lambda x: (x - x.min()) / (x.max() - x.min()))
        expr = expr.fillna(0.0).apply(
            lambda x: (x - x.min()) / (x.max() - x.min()))

        if len(mut_levels) > 1 or mut_levels[0] != 'Gene':
            if 'Gene' in mut_levels:
                scale_lvl = mut_levels.index('Gene') + 1
            else:
                scale_lvl = 0
 
            mut_levels.insert(scale_lvl, 'Scale')
            mut_levels.insert(scale_lvl + 1, 'Copy')
            variants['Scale'] = 'Point'
            copy_df['Scale'] = 'Copy'

        super().__init__(pd.concat([expr, beataml_expr], sort=True),
                         pd.concat([variants, copy_df], sort=True),
                         mut_genes, mut_levels, top_genes, samp_cutoff,
                         cv_prop, cv_seed)

