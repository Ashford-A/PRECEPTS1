
from ..data.expression import get_expr_toil
from dryadic.features.cohorts import BaseMutationCohort
from dryadic.features.cohorts.utils import get_gencode, drop_duplicate_genes

import os
import pandas as pd


def load_ccle_samps(ccle_dir):
    return pd.read_csv(os.path.join(ccle_dir, "data_clinical_sample.txt"),
                       sep='\t', index_col=0, comment='#')


def load_ccle_expression(ccle_dir, expr_source, **expr_args):
    if expr_source == 'microarray':
        expr_mat = pd.read_csv(
            os.path.join(ccle_dir, "data_expression_median.txt"),
            sep='\t', index_col=0
            ).transpose()[1:].fillna(0.0)

    elif expr_source == 'toil':
        expr_mat = get_expr_toil(cohort, expr_args['expr_dir'],
                                 expr_args['collapse_txs'])

    else:
        raise ValueError("Unrecognized source of CCLE expression "
                         "data `{}` !".format(expr_source))

    return expr_mat


def load_ccle_variants(ccle_dir, var_source='default'):
    if var_source == 'default':
        use_cols = [0, 5, 9, 16, 37, 39, 40]
        use_names = ['Gene', 'Start', 'Form', 'Sample',
                     'Nucleo', 'Protein', 'Transcript']

        mut_df = pd.read_csv(
            os.path.join(ccle_dir, "data_mutations_mskcc.txt"),
            names=use_names, usecols=use_cols, engine='python', sep='\t',
            header=None, comment='#', skiprows=2
            )

    elif var_source == 'profiles':
        mut_df = pd.read_csv(
            os.path.join(ccle_dir, "mutationalProfiles",
                         "Data", "somaticMutations_incNC.txt"),
            engine='python', sep='\t'
            )

        mut_df['Exon'] = ['.' if pd.isnull(exn) else int(exn.split('exon')[1])
                          for exn in mut_df.exon]

        mut_df['alt_count'] = pd.to_numeric(
            (mut_df.reads * mut_df.vaf).round(0), downcast='integer')
        mut_df['ref_count'] = pd.to_numeric(
            (mut_df.reads * (1 - mut_df.vaf)).round(0), downcast='integer')

        mut_df = mut_df.rename(columns={
            'sample': 'Sample', 'gene': 'Gene', 'exon': 'Exon'})

        mut_df.Form = mut_df.mutationType.map({
            'missense SNV': 'Missense_Mutation',
            'silent SNV': 'Silent',
            'frameshift indel': 'Frame_Shift_Ins',
            'nonsense SNV': 'Nonsense_Mutation',
            'inframe indel': 'In_Frame_Del',
            'stoploss': 'Nonstop_Mutation'
            })

    else:
        raise ValueError("Unrecognized source of METABRIC variant "
                         "data `{}` !".format(var_source))

    return mut_df


def load_ccle_copies(ccle_dir):
    return pd.read_csv(os.path.join(ccle_dir, "data_CNA.txt"),
                       sep='\t', index_col=0).transpose()[1:]


class CellLineCohort(BaseMutationCohort):

    def __init__(self,
                 mut_levels, mut_genes, expr_source, ccle_dir, annot_file,
                 domain_dir=None, cv_seed=None, test_prop=0,
                 leaf_annot=('Nucleo', ), **coh_args):
        self.cohort = 'CCLE'

        samp_data = load_ccle_samps(ccle_dir)
        expr = load_ccle_expression(ccle_dir, expr_source, **coh_args)
        muts = load_ccle_variants(ccle_dir)
        copies = load_ccle_copies(ccle_dir)

        expr.index = [samp_data.index[samp_data.SAMPLE_ID == smp][0]
                      for smp in expr.index]
        copies.index = [samp_data.index[samp_data.SAMPLE_ID == smp][0]
                        for smp in copies.index]
        muts.Sample = [samp_data.index[samp_data.SAMPLE_ID == smp][0]
                       for smp in muts.Sample]

        use_samps = set(expr.index) & set(copies.index)
        expr = drop_duplicate_genes(expr.loc[use_samps])
        muts = muts.loc[muts.Sample.isin(use_samps)]
        annot_data = get_gencode(annot_file, ['transcript', 'exon'])

        self.gene_annot = {at['gene_name']: {**{'Ens': ens}, **at}
                           for ens, at in annot_data.items()
                           if at['gene_name'] in set(expr.columns)}

        expr = expr.loc[:, expr.columns.isin(self.gene_annot)]
        muts = muts.loc[muts.Gene.isin(self.gene_annot.keys())]
        muts['Scale'] = 'Point'

        copies = copies.loc[use_samps, copies.columns.isin(self.gene_annot)]
        copy_df = pd.DataFrame(copies.stack()).reset_index()
        copy_df.columns = ['Sample', 'Gene', 'Copy']

        copy_df = copy_df.loc[(copy_df.Copy != 0)]
        copy_df.Copy = copy_df.Copy.map({-2: 'DeepDel', -1: 'ShalDel',
                                         1: 'ShalGain', 2: 'DeepGain'})
        copy_df['Scale'] = 'Copy'

        muts['Exon'] = [
            tuple(exn_no)[0] if len(exn_no) == 1 else '.'
            for exn_no in [{
                exn['number'] for exn in self.gene_annot[
                    mut.Gene]['Transcripts'][mut.Transcript]['Exons']
                    if exn['Start'] <= mut.Start <= exn['End']
                }
                if ('Transcripts' in self.gene_annot[mut.Gene]
                    and mut.Transcript in self.gene_annot[
                        mut.Gene]['Transcripts'])
                else set() for mut in muts.itertuples(index=False)
                ]
            ]

        for i in range(len(mut_levels)):
            if 'Scale' not in mut_levels[i]:
                if 'Gene' in mut_levels[i]:
                    scale_lvl = mut_levels[i].index('Gene') + 1
                else:
                    scale_lvl = 0

                mut_levels[i].insert(scale_lvl, 'Scale')
                mut_levels[i].insert(scale_lvl + 1, 'Copy')

        super().__init__(expr, pd.concat([muts, copy_df], sort=True),
                         mut_levels, mut_genes, domain_dir,
                         leaf_annot, cv_seed, test_prop)

