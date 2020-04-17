
from dryadic.features.cohorts import BaseMutationCohort
from dryadic.features.cohorts.utils import (
    get_gencode, log_norm, drop_duplicate_genes)

import os
import numpy as np
import pandas as pd


def load_beat_expression(baml_dir):
    expr_file = os.path.join(baml_dir, "RNAseq", "merged",
                             "baml_rnaseq_ch1-22_0.02tpm.tsv")

    return log_norm(pd.read_csv(expr_file, sep='\t', index_col=0).fillna(0.0))


def load_beat_variants(syn, **var_args):
    field_dict = (
        ('Chr', 1), ('Start', 2), ('End', 3), ('RefAllele', 4),
        ('TumorAllele', 5), ('tot_reads', 8), ('alt_reads', 9),
        ('Consequence', 15), ('Gene', 17), ('Exon', 22),
        ('HGVSc', 23), ('HGVSp', 24), ('Variant_Class', 31), ('Sample', 38)
        )

    if 'mut_fields' not in var_args or var_args['mut_fields'] is None:
        use_fields, use_cols = tuple(zip(*field_dict))

    else:
        if 'Transcript' in var_args['mut_fields']:
            var_args['mut_fields'] = set(var_args['mut_fields']) | {'HGVSc'}
        if 'Form' in var_args['mut_fields']:
            var_args['mut_fields'] = set(
                var_args['mut_fields']) | {'Consequence', 'Variant_Class'}

        use_fields, use_cols = tuple(zip(*[
            (name, col) for name, col in field_dict
            if name in {'Sample'} | set(var_args['mut_fields'])
            ]))

    var_data = pd.read_csv(syn.get('syn18683049').path,
                           engine='c', dtype='object', sep='\t', header=None,
                           skiprows=1, usecols=use_cols, names=use_fields)

    if 'Transcript' in use_fields:
        var_data['Transcript'] = var_data.HGVSc.str.split('\\.[0-9]+:').apply(
            lambda x: '.' if isinstance(x, float) else x[0])

    if 'Form' in use_fields:
        var_data['Form'] = variants.Consequence.map({
            'missense_variant': 'Missense_Mutation',
            'frameshift_variant': 'frameshift_variant',
            'inframe_deletion': 'In_Frame_Del',
            'inframe_insertion': 'In_Frame_Ins',
            'stop_gained': 'Nonsense_Mutation',
            'start_lost': 'Translation_Start_Site',
            'protein_altering_variant': 'Nonsense_Mutation',
            'stop_lost': 'Nonstop_Mutation',
            'internal_tandem_duplication': 'ITD',
            'splice_acceptor_variant': 'Splice_Site',
            'splice_donor_variant': 'Splice_Site',
            })

        var_data.loc[(var_data.Form == 'frameshift_variant')
                   & (var_data.Variant_Class == 'insertion'),
                   'Form'] = 'Frame_Shift_Ins'
        var_data.loc[(var_data.Form == 'frameshift_variant')
                   & (var_data.Variant_Class == 'deletion'),
                   'Form'] = 'Frame_Shift_Del'

    if 'ref_count' in use_fields:
        var_data['ref_count'] = var_data.total_reads - var_data.alt_count

    return var_data


def process_input_datasets(baml_dir, annot_dir, syn, **data_args):

    samp_data = pd.read_csv(os.path.join(baml_dir, "VarCalls",
                                         "TableS12_WES_samples.tsv"),
                            sep='\t')
    expr = load_beat_expression(baml_dir)

    annot_file = os.path.join(annot_dir, "gencode.v19.annotation.gtf.gz")
    if 'annot_fields' in data_args:
        annot_data = get_gencode(annot_file, data_args['annot_fields'])
    else:
        annot_data = get_gencode(annot_file)

    annot_dict = {at['gene_name']: {**{'Ens': ens}, **at}
                  for ens, at in annot_data.items()
                  if ens in set(expr.columns)}

    # TODO: incorporate supplemental mutation data, eg. laboratory-based
    # data for FLT3 ITDs found here:
    # https://www.nature.com/articles/s41586-018-0623-z
    variants = load_beat_variants(syn, **data_args)
    variants['Sample'] = ['pid{}'.format(pid) for pid in variants.Sample.values]
    samp_data['Sample'] = ['pid{}'.format(pid) for pid in samp_data.patientId]
    use_samps = set(expr.index) & set(samp_data.Sample)

    expr = expr.loc[use_samps, expr.columns.isin(annot_data)]
    expr_data = drop_duplicate_genes(expr.rename(
        columns={gn: annot_data[gn]['gene_name'] for gn in expr.columns}))

    variants = variants.loc[variants.Sample.isin(use_samps)
                            & variants.Gene.isin(annot_dict)]
    variants['Scale'] = 'Point'

    if 'Gene' in variants:
        variants['Strand'] = [annot_dict[gn]['Strand']
                              for gn in variants.Gene.values]

    return expr_data, variants, annot_dict


class BeatAmlCohort(BaseMutationCohort):

    def __init__(self,
                 baml_dir=None, annot_dir=None, syn=None, expr_data=None,
                 mut_data=None, annot_data=None, mut_levels=None,
                 mut_genes=None, collapse_txs=True,
                 leaf_annot=('ref_count', 'alt_count'),
                 cv_seed=None, test_prop=0, **coh_args):
        self.cohort = 'beatAML'

        expr_data, mut_data, gene_annot = process_input_datasets(
            baml_dir, annot_dir, syn, collapse_txs, **coh_args)
 
        super().__init__(expr_data, mut_data, mut_levels, mut_genes,
                         gene_annot, leaf_annot, cv_seed, test_prop)

