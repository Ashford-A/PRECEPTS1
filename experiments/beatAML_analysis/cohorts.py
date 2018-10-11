
from dryadic.features.cohorts import BaseMutationCohort
from dryadic.features.cohorts.utils import get_gencode, log_norm
from ...features.cohorts.tcga import add_variant_data

import pandas as pd
import os
from operator import itemgetter


class CancerCohort(BaseMutationCohort):
 
    def __init__(self,
                 mut_genes, mut_levels, toil_dir, sample_data, tx_map,
                 syn, copy_dir, annot_file, top_genes=None, samp_cutoff=25,
                 cv_prop=0.75, cv_seed=None, **coh_args):
        tcga_expr = pd.read_csv(
            os.path.join(toil_dir, 'TCGA', 'TCGA_LAML_tpm.tsv.gz'),
            sep='\t'
            )

        tcga_expr.index = tcga_expr.iloc[
            :, 0].str.split('|').apply(itemgetter(0))
        tcga_expr = tcga_expr.iloc[:, 1:]

        id_map = pd.read_csv(os.path.join(toil_dir, 'TCGA_ID_MAP.csv'),
                             sep=',', index_col=0)
        id_map = id_map.loc[id_map['Disease'] == 'LAML']
        tcga_expr.columns = id_map.loc[tcga_expr.columns,
                                       'AliquotBarcode'].values

        tx_annot = pd.read_csv(tx_map, sep='\t', index_col=0)
        tcga_expr.index.name = 'Transcript'
        tcga_expr = tcga_expr.transpose().loc[:, [tx in tx_annot.index
                                                  for tx in tcga_expr.index]]

        tcga_expr.columns = pd.MultiIndex.from_arrays(
            [tx_annot.loc[tcga_expr.columns, 'gene'], tcga_expr.columns],
            names=['Gene', 'Transcript']
            )
    
        tcga_expr.sort_index(axis=1, level=['Gene'], inplace=True)
        annot_data = get_gencode(annot_file)

        self.gene_annot = {at['gene_name']: {**{'Ens': ens}, **at}
                           for ens, at in annot_data.items()
                           if at['gene_name'] in tcga_expr.columns}

        expr, variants, copy_df = add_variant_data(
            cohort='LAML', var_source='mc3', copy_source='Firehose', syn=syn,
            expr=tcga_expr, copy_dir=copy_dir, gene_annot=self.gene_annot,
            )

        beataml_expr = pd.read_csv(
            os.path.join(sample_data, 'matrices', 'CTD2_TPM_transcript.tsv'),
            sep='\t', index_col=0
            )

        use_genes = sorted(set(beataml_expr.index)
                           & set(expr.columns.get_level_values('Transcript')))
        expr = log_norm(expr.loc[:, (slice(None), use_genes)])

        beataml_expr = log_norm(beataml_expr.transpose()).loc[:, use_genes]
        beataml_expr = beataml_expr.loc[
            :, expr.columns.get_level_values('Transcript')]

        beataml_expr.columns = pd.MultiIndex.from_arrays(
            [expr.columns.get_level_values('Gene'), beataml_expr.columns],
            names=['Gene', 'Transcript']
            )
    
        beataml_expr = beataml_expr.apply(
            lambda x: (x - x.min()) / (x.max() - x.min())).fillna(0.0)
        expr = expr.apply(
            lambda x: (x - x.min()) / (x.max() - x.min())).fillna(0.0)

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

