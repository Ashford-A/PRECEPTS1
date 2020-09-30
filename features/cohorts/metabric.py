
from dryadic.features.cohorts import BaseMutationCohort
from dryadic.features.cohorts.utils import get_gencode, drop_duplicate_genes

import os
import pandas as pd


def load_metabric_samps(metabric_dir):
    return pd.read_csv(os.path.join(metabric_dir, "data_clinical_sample.txt"),
                       sep='\t', index_col=0, comment='#')


def load_metabric_expression(metabric_dir, expr_source='microarray'):
    if expr_source == 'microarray':
        expr_file = os.path.join(metabric_dir, "data_expression_median.txt")
    else:
        raise ValueError("Unrecognized source of METABRIC expression "
                         "data `{}` !".format(expr_source))

    return pd.read_csv(expr_file,
                       sep='\t', index_col=0).transpose()[1:].fillna(0.0)


def load_metabric_variants(metabric_dir, var_source='default', **var_args):
    if var_source == 'default':
        field_dict = (
            ('Gene', 0), ('Chr', 4), ('Start', 5), ('End', 6), ('Strand', 7),
            ('Form', 9), ('RefAllele', 11), ('TumorAllele', 13),
            ('Sample', 16), ('HGVSc', 37), ('HGVSp', 38), ('Protein', 39),
            ('Transcript', 40), ('Position', 42)
            )

        if 'mut_fields' not in var_args or var_args['mut_fields'] is None:
             use_fields, use_cols = tuple(zip(*field_dict))

        else:
            use_fields, use_cols = tuple(zip(*[
                (name, col) for name, col in field_dict
                if name in {'Sample'} | set(var_args['mut_fields'])
                ]))

        mut_df = pd.read_csv(
            os.path.join(metabric_dir, "data_mutations_mskcc.txt"),
            engine='c', dtype='object', sep='\t', header=None,
            usecols=use_cols, names=use_fields, comment='#', skiprows=2
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


def choose_subtypes(samp_data, use_types):
    if use_types == 'Basal':
        sub_samps = set(samp_data[
            (samp_data.HER2_STATUS == 'Negative')
            & (samp_data.ER_STATUS == 'Negative')
            & (samp_data.PR_STATUS == 'Negative')
            ].SAMPLE_ID)

    elif use_types == 'LumA':
        sub_samps = set(samp_data[
            (samp_data.HER2_STATUS == 'Negative')
            & ((samp_data.ER_STATUS == 'Positive')
               | (samp_data.PR_STATUS == 'Positive'))
            ].SAMPLE_ID)

    elif use_types == 'LumB':
        sub_samps = set(samp_data[
            (samp_data.HER2_STATUS == 'Positive')
            & ((samp_data.ER_STATUS == 'Positive')
               | (samp_data.PR_STATUS == 'Positive'))
            ].SAMPLE_ID)

    elif use_types == 'Her2':
        sub_samps = set(samp_data[
            (samp_data.HER2_STATUS == 'Positive')
            & (samp_data.ER_STATUS == 'Negative')
            & (samp_data.PR_STATUS == 'Negative')
            ].SAMPLE_ID)

    elif use_types == 'luminal':
        sub_samps = set(samp_data[
            (samp_data.ER_STATUS == 'Positive')
            | (samp_data.PR_STATUS == 'Positive')
            ].SAMPLE_ID)

    elif use_types == 'nonbasal':
        sub_samps = set(samp_data[
            (samp_data.HER2_STATUS == 'Positive')
            | (samp_data.ER_STATUS == 'Positive')
            | (samp_data.PR_STATUS == 'Positive')
            ].SAMPLE_ID)

    else:
        raise ValueError("Unrecognized molecular subtype `{}` for the "
                         "METABRIC cohort!".format(use_types))

    return sub_samps


def process_input_datasets(metabric_dir, annot_dir, use_types=None,
                           **data_args):

    samp_data = load_metabric_samps(metabric_dir)
    expr = drop_duplicate_genes(load_metabric_expression(metabric_dir))

    use_samps = set(samp_data.SAMPLE_ID[
        (samp_data.CANCER_TYPE == 'Breast Cancer')
        & (samp_data.CANCER_TYPE_DETAILED
           == 'Breast Invasive Ductal Carcinoma')
        ]) & set(expr.index)

    annot_file = os.path.join(annot_dir, "gencode.v19.annotation.gtf.gz")
    if 'annot_fields' in data_args:
        annot_data = get_gencode(annot_file, data_args['annot_fields'])
    else:
        annot_data = get_gencode(annot_file)

    annot_dict = {at['gene_name']: {**{'Ens': ens}, **at}
                  for ens, at in annot_data.items()
                  if at['gene_name'] in set(expr.columns)}

    variants = load_metabric_variants(metabric_dir)
    copy_df = load_metabric_copies(metabric_dir)

    use_samps &= set(copy_df.index)
    with open(os.path.join(metabric_dir,
                           "data_mutations_mskcc.txt"), 'r') as f:
        use_samps &= set(f.readline().split(
            "#Sequenced_Samples: ")[1].split('\t')[0].split(' '))

    if use_types is not None:
        use_samps &= choose_subtypes(samp_data, use_types)

    expr_data = expr.loc[use_samps, expr.columns.isin(annot_dict)]
    variants = variants.loc[variants.Sample.isin(use_samps)
                            & variants.Gene.isin(annot_dict)]

    copy_df = copy_df.loc[use_samps, copy_df.columns.isin(annot_dict)]
    copy_df = pd.DataFrame(copy_df.stack()).reset_index()
    copy_df.columns = ['Sample', 'Gene', 'Copy']

    copy_df = copy_df.loc[(copy_df.Copy != 0)]
    copy_df.Copy = copy_df.Copy.map({-2: 'DeepDel', -1: 'ShalDel',
                                     1: 'ShalGain', 2: 'DeepGain'})

    return expr_data, variants, copy_df, annot_dict


class MetabricCohort(BaseMutationCohort):

    def __init__(self,
                 metabric_dir, annot_dir, expr_data=None, mut_data=None,
                 annot_data=None, mut_levels=None, mut_genes=None,
                 leaf_annot=('Nucleo', ), cv_seed=None, test_prop=0,
                 **coh_args):
        self.cohort = 'METABRIC'

        if expr_data is None or mut_data is None or annot_data is None:
            expr_data, mut_data, self.annot_data = process_input_datasets(
                metabric_dir, annot_dir, **coh_args)

        else:
            self.annot_data = annot_data

        super().__init__(expr, mut_data, mut_levels, mut_genes,
                         leaf_annot, cv_seed, test_prop)

