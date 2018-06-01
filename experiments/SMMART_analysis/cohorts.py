
from ...features.cohorts.mut import BaseMutationCohort
from .utils import load_patient_expression
from ...features.cohorts.tcga import match_tcga_samples

from ...features.data.expression import get_expr_toil
from ...features.data.variants import get_variants_mc3, get_variants_firehose
from ...features.data.copies import get_copies_firehose
from ...features.data.annot import get_gencode

import numpy as np
import pandas as pd

from functools import reduce
from operator import and_


class CancerCohort(BaseMutationCohort):
 
    def __init__(self,
                 cancer, mut_genes, mut_levels,
                 tcga_dir, patient_dir, var_source='mc3', copy_source=None,
                 top_genes=None, samp_cutoff=25, collapse_txs=False,
                 cv_prop=0.75, cv_seed=None, **coh_args):

        cancer_keys = {'BRCA': '1', 'PRAD': '2', 'PAAD': '3', 'LAML': '4'}
        if cancer not in cancer_keys:
            raise ValueError(
                "SMMART samples are only available for breast (BRCA), "
                "prostate (PRAD), pancreatic (PAAD), and AML cancers!"
                )

        tcga_expr = get_expr_toil(cohort=cancer, data_dir=tcga_dir,
                                  collapse_txs=False)

        if var_source is None:
            var_source = expr_source

        if var_source == 'mc3':
            variants = get_variants_mc3(coh_args['syn'])

        elif var_source == 'Firehose':
            variants = get_variants_firehose(cohort, coh_args['var_dir'])

        else:
            raise ValueError("Unrecognized source of variant data!")

        if copy_source == 'Firehose':
            copy_data = get_copies_firehose(cohort, coh_args['copy_dir'])

            if 'Gene' in mut_levels:
                copy_lvl = mut_levels[mut_levels.index('Gene') + 1]
            else:
                copy_lvl = mut_levels[0]

            # reshapes the matrix of CNA values into the same long format
            # mutation data is represented in
            copy_lvl = copy_lvl.split('_')[0]
            copy_df = pd.DataFrame(copy_data.stack())
            copy_df = copy_df.reset_index(level=copy_df.index.names)
            copy_df.columns = ['Sample', 'Gene', copy_lvl]

            # removes CNA values corresponding to an absence of a variant
            copy_df = copy_df.loc[copy_df[copy_lvl] != 0, :]

            # maps CNA integer values to their descriptions, appends
            # CNA data to the mutation data
            copy_df[copy_lvl] = copy_df[copy_lvl].map(
                {-2: 'HomDel', -1: 'HetDel', 1: 'HetGain', 2: 'HomGain'})
            variants = pd.concat([variants, copy_df])

        elif copy_source is not None:
            raise ValueError("Unrecognized source of CNA data!")

        smrt_dict = {
            (tuple(patient[0].split('-')), tuple(patient[1].split('_'))): expr
            for patient, expr in load_patient_expression(patient_dir).items()
            }
        smrt_dict = {patient: expr for patient, expr in smrt_dict.items()
                     if patient[0][1][0] == cancer_keys[cancer]}

        if len(set(patient[0][0] for patient in smrt_dict)) == 1:
            smrt_dict = {(patient[0][1], patient[1]): expr
                         for patient, expr in smrt_dict.items()}

        annot_rmv = [len(set(patient[1][i] for patient in smrt_dict)) == 1
                     for i in range(3)]
        smrt_dict = {
            '{} --- {}'.format(
                patient[0], '_'.join([x for i, x in enumerate(patient[1][:-1])
                                      if i not in annot_rmv])
                ): expr
            for patient, expr in smrt_dict.items()
            }

        smrt_txs = reduce(and_, [expr.keys() for expr in smrt_dict.values()])
        smrt_expr = pd.DataFrame.from_dict(
            {patient: {tx: expr[tx, gn] for tx, gn in smrt_txs
                       if tx in tcga_expr.columns.levels[1]}
             for patient, expr in smrt_dict.items()},
            orient='index'
            )

        matched_samps = match_tcga_samples(tcga_expr.index,
                                           variants['Sample'])
        matched_samps += [(samp, (samp, None)) for samp in smrt_expr.index]

        tcga_expr = tcga_expr.loc[:, [tx in smrt_expr.columns
                                      for gn, tx in tcga_expr.columns]]
        tcga_expr.columns = tcga_expr.columns.remove_unused_levels()

        tx_vals = tcga_expr.columns.get_level_values('Transcript')
        smrt_expr = smrt_expr.iloc[:, [smrt_expr.columns.get_loc(tx)
                                       for tx in tx_vals]]

        smrt_expr.columns = pd.MultiIndex.from_arrays(
            [tcga_expr.columns.get_level_values('Gene'), smrt_expr.columns])
        expr = pd.concat([tcga_expr, smrt_expr])
        gn_vals = expr.columns.get_level_values('Gene')

        # reduce transcription-level expression values into gene-level values
        if collapse_txs:
            expr = np.log2(expr.rpow(2).subtract(0.001).groupby(
                level=['Gene'], axis=1).sum().add(0.001))

        gene_annot = {at['gene_name']: {**{'Ens': ens}, **at}
                      for ens, at in get_gencode().items()
                      if at['gene_name'] in gn_vals}
        self.cohort = cancer

        super().__init__(expr, variants, matched_samps, gene_annot, mut_genes,
                         mut_levels, top_genes, samp_cutoff, cv_prop, cv_seed)

