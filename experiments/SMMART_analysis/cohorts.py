
from ...features.cohorts.mut import BaseMutationCohort
from .utils import load_patient_expression, load_patient_mutations
from ...features.cohorts.tcga import match_tcga_samples

from ...features.data.expression import get_expr_toil
from ...features.data.variants import get_variants_mc3, get_variants_firehose
from ...features.data.copies import get_copies_firehose
from ...features.data.annot import get_gencode

import numpy as np
import pandas as pd
import os

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

        smrt_expr = load_patient_expression(patient_dir)
        smrt_mut = load_patient_mutations(patient_dir)

        meta_data = pd.read_csv(
            open(os.path.join(patient_dir, '..', 'metadata', 'metadata.tsv'),
                 'r'),
            sep='\t'
            )

        match_dict = dict()
        for patient, out_dir in smrt_expr.keys():
            pt_data = meta_data.loc[meta_data['patient_id'] == patient, :]
            lib_match = np.array(
                [lib_id in out_dir for lib_id in pt_data['library_id']])

            bems_id = pt_data['sample_bems_id'][lib_match]
            mut_id = np.unique(pt_data['library_id'][
                pt_data['sample_bems_id'].isin(bems_id) & ~lib_match])

            if len(mut_id) == 1:
                mut_key = [(pt, out_fl) for pt, out_fl in smrt_mut.keys()
                           if (pt == patient and mut_id[0] in str(out_fl)
                               and '_sorted.maf' not in str(out_fl))]

                if len(mut_key) == 1:
                    match_dict[mut_key[0]] = patient, out_dir

        smrt_mut = {match_dict[mut_key]: mut_vals
                    for mut_key, mut_vals in smrt_mut.items()
                    if mut_key in match_dict}

        pt_tags = {patient: (patient[0].split('-'), patient[1].split('_'))
                   for patient in smrt_expr}
        pt_tags = {patient: tag for patient, tag in pt_tags.items()
                   if tag[0][1][0] == cancer_keys[cancer]}

        if len(set(tag[0][0] for pt, tag in pt_tags)) == 1:
            pt_tags = {pt: (tag[0][1], tag[1]) for pt, tag in pt_tags.items()}

        annot_rmv = [
            len(set(lbls[i] for pt, (tag, lbls) in pt_tags.items())) == 1
            for i in range(3)
            ]

        pt_tags = {
            pt: '{} --- {}'.format(
                tag, '_'.join([lbl for rmv, lbl in zip(annot_rmv, lbls[:3])
                               if not rmv])
                )
            for pt, (tag, lbls) in pt_tags.items()
            }

        smrt_expr = {pt_tags[pt]: expr for pt, expr in smrt_expr.items()
                     if pt in pt_tags}
        smrt_mut = {pt_tags[pt]: mut for pt, mut in smrt_mut.items()
                    if pt in pt_tags}

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

        smrt_txs = reduce(and_, [expr.keys() for expr in smrt_expr.values()])
        smrt_expr = pd.DataFrame.from_dict(
            {patient: {tx: expr[tx, gn] for tx, gn in smrt_txs
                       if tx in tcga_expr.columns.levels[1]}
             for patient, expr in smrt_expr.items()},
            orient='index'
            )

        matched_samps = match_tcga_samples(tcga_expr.index,
                                           variants['Sample'])
        matched_samps += [(samp, (samp, samp)) for samp in smrt_expr.index]

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

        smrt_mut = pd.concat(
            [muts.loc[:, ['Hugo_Symbol', 'Variant_Classification']]\
                .drop_duplicates().assign(Sample=pt)
             for pt, muts in smrt_mut.items()]
            )

        smrt_mut.columns = ['Gene', 'Form', 'Sample']
        variants = pd.concat([variants, smrt_mut])

        gene_annot = {at['gene_name']: {**{'Ens': ens}, **at}
                      for ens, at in get_gencode().items()
                      if at['gene_name'] in gn_vals}
        self.cohort = cancer

        super().__init__(expr, variants, matched_samps, gene_annot, mut_genes,
                         mut_levels, top_genes, samp_cutoff, cv_prop, cv_seed)

