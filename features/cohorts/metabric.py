
from dryadic.features.cohorts import BaseMutationCohort
from dryadic.features.cohorts.utils import get_gencode, drop_duplicate_genes

import os
import pandas as pd


def load_metabric_expression(metabric_dir, expr_source='microarray'):
    if expr_source == 'microarray':
        expr_file = os.path.join(metabric_dir, "data_expression_median.txt")
    else:
        raise ValueError("Unrecognized source of METABRIC expression "
                         "data `{}` !".format(expr_source))

    return pd.read_csv(expr_file,
                       sep='\t', index_col=0).transpose()[1:].fillna(0.0)


def load_metabric_variants(metabric_dir, var_source='default'):
    if var_source == 'default':
        use_cols = [0, 5, 9, 16, 37, 39, 40]
        use_names = ['Gene', 'Start', 'Form', 'Sample',
                     'Nucleo', 'Protein', 'Transcript']

        mut_df = pd.read_csv(
            os.path.join(metabric_dir, "data_mutations_mskcc.txt"),
            names=use_names, usecols=use_cols, engine='python', sep='\t',
            header=None, comment='#', skiprows=2
            )

    elif var_source == 'profiles':
        mut_df = pd.read_csv(
            os.path.join(metabric_dir, "mutationalProfiles",
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


def load_metabric_copies(metabric_dir):
    return pd.read_csv(os.path.join(metabric_dir, "data_CNA.txt"),
                       sep='\t', index_col=0).transpose()[1:]


class MetabricCohort(BaseMutationCohort):

    def __init__(self,
                 mut_levels, mut_genes, metabric_dir, annot_file,
                 domain_dir=None, cv_seed=None, test_prop=0,
                 leaf_annot=('Nucleo', ), **coh_args):
        self.cohort = 'METABRIC'

        expr = load_metabric_expression(metabric_dir)
        muts = load_metabric_variants(metabric_dir)
        copies = load_metabric_copies(metabric_dir)

        samp_data = pd.read_csv(
            os.path.join(metabric_dir, "data_clinical_sample.txt"),
            sep='\t', index_col=0, comment='#'
            )

        use_samps = set(samp_data.SAMPLE_ID[
            (samp_data.CANCER_TYPE == 'Breast Cancer')
            & (samp_data.CANCER_TYPE_DETAILED
               == 'Breast Invasive Ductal Carcinoma')
            ]) & set(expr.index)

        use_samps &= set(copies.index)
        with open(os.path.join(metabric_dir,
                               "data_mutations_mskcc.txt"), 'r') as f:
            use_samps &= set(f.readline().split(
                "#Sequenced_Samples: ")[1].split('\t')[0].split(' '))

        if 'use_types' in coh_args and coh_args['use_types'] is not None:
            if coh_args['use_types'] == 'Basal':
                use_samps &= set(
                    samp_data[(samp_data.HER2_STATUS == 'Negative')
                              & (samp_data.ER_STATUS == 'Negative')
                              & (samp_data.PR_STATUS == 'Negative')].SAMPLE_ID
                    )

            elif coh_args['use_types'] == 'LumA':
                use_samps &= set(samp_data[
                    (samp_data.HER2_STATUS == 'Negative')
                    & ((samp_data.ER_STATUS == 'Positive')
                       | (samp_data.PR_STATUS == 'Positive'))].SAMPLE_ID)

            elif coh_args['use_types'] == 'LumB':
                use_samps &= set(samp_data[
                    (samp_data.HER2_STATUS == 'Positive')
                    & ((samp_data.ER_STATUS == 'Positive')
                       | (samp_data.PR_STATUS == 'Positive'))].SAMPLE_ID)

            elif coh_args['use_types'] == 'Her2':
                use_samps &= set(
                    samp_data[(samp_data.HER2_STATUS == 'Positive')
                              & (samp_data.ER_STATUS == 'Negative')
                              & (samp_data.PR_STATUS == 'Negative')].SAMPLE_ID
                    )

            elif coh_args['use_types'] == 'luminal':
                use_samps &= set(
                    samp_data[(samp_data.ER_STATUS == 'Positive')
                              | (samp_data.PR_STATUS == 'Positive')].SAMPLE_ID
                    )

            elif coh_args['use_types'] == 'nonbasal':
                use_samps &= set(
                    samp_data[(samp_data.HER2_STATUS == 'Positive')
                              | (samp_data.ER_STATUS == 'Positive')
                              | (samp_data.PR_STATUS == 'Positive')].SAMPLE_ID
                    )

            else:
                raise ValueError(
                    "Unrecognized molecular subtype `{}` for the METABRIC "
                    "cohort!".format(coh_args['use_types'])
                    )

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
                } for mut in muts.itertuples(index=False)]
            ]

        for i in range(len(mut_levels)):
            if 'Gene' in mut_levels[i]:
                scale_lvl = mut_levels[i].index('Gene') + 1
            else:
                scale_lvl = 0

            mut_levels[i].insert(scale_lvl, 'Scale')
            mut_levels[i].insert(scale_lvl + 1, 'Copy')

        super().__init__(expr, pd.concat([muts, copy_df], sort=True),
                         mut_levels, mut_genes, domain_dir,
                         leaf_annot, cv_seed, test_prop)

