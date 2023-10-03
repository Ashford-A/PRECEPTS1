'''
Andrew Ashford, Pathways + Omics group, OHSU, 9/12/2023
This script is a copy of Michal Grzadkowski's "dryads-research/features/cohorts/beatAML.py"
code, modified to load in the BeatAML bulk RNA-seq waves 1-4 data. The data consists of 
RNA-seq, WES calls, and sample data. At present, the data is all stored on Exacloud, but to
make the reproducible results for the publication later, we likely will need to house some 
of the input data on a Synpase repository, which will require specific permissions to 
access input data.
'''

from dryadic.features.cohorts import BaseMutationCohort
from dryadic.features.cohorts.utils import (
    get_gencode, log_norm, drop_duplicate_genes)

import os
import numpy as np
import pandas as pd

def load_beat_expression(baml_dir):
    # Using tpms normalized input RNA-seq counts.
    # NOTE: This dataframe of RNAseq counts was transposed such that genes are columns and samples are rows.
    expr_file = os.path.join(baml_dir, "beataml_waves1to4_norm_transposed.tsv")

    return log_norm(pd.read_csv(expr_file, sep='\t', index_col=0).fillna(0.0))


# NOTE: "syn" variable no longer actually means accessing a Synergy repo file.. file is
# accessed locally in the baml waves 1-4 dir. Called "beataml_wes_wv1to4_mutations_v4.tsv".
# Second file that was on Synapse client likely doesn't need to be used.. can probably
# get all the WES mutation data directly from the previously specified local file.
# 
# The file "beataml_wes_wv1to4_mutations_v4.tsv" was added to so now is called:
# "beataml_wes_wv1to4_mutations_v4_exon_mut_loc_and_variant_class_added.tsv"
def load_beat_variants(syn, **var_args):
    field_dict = (
        ('Chr', 0), ('Start', 1), ('End', 2), ('RefAllele', 3),
        ('TumorAllele', 4), ('Genotyper', 5), ('tot_reads', 10),
        ('alt_reads', 11), ('Consequence', 16), ('Sample', 8), 
        ('HGVSc', 21), ('HGVSp', 22), ('Gene', 19), ('Exon', 33),
        ('Variant_Class', 34)
        #('Exon', 22) # Don't have Exon in spreadsheet
        #('Variant_Class', 31) # Don't have simplified variant calss in spreadsheet, just have "consequence" of muts
        
        ### Added both Exon and Variant class using Funcotator annotation tool. Column names in spreadsheet are
        ### named "Gencode_34_transcriptExon" and "Gencode_34_variantType". Had to make a "vcf" file from the mutation
        ### calls Excel spreadsheet in order to annotate/run.
        
        ### Changed 'Sample' column to the 8th column in the varient spreadsheet. I determined that the RNA-seq data
        ### is using the "original_id" as identifiers.
        
        ### NOTE ### Need to transpose the RNAseq data so that the identifiers are rows and genes are columns
        )

    if 'mut_fields' not in var_args or var_args['mut_fields'] is None:
        use_fields, use_cols = tuple(zip(*field_dict))

    else:
        if 'Transcript' in var_args['mut_fields']:
            var_args['mut_fields'] = set(var_args['mut_fields']) | {'HGVSc'}
        if 'Form' in var_args['mut_fields']:
            '''
            var_args['mut_fields'] = set(
                var_args['mut_fields']) | {'Consequence', 'Variant_Class'}
            '''
            var_args['mut_fields'] = set(
                var_args['mut_fields']) | {'Consequence', 'Variant_Class'}

        use_fields, use_cols = tuple(zip(*[
            (name, col) for name, col in field_dict
            if name in {'Sample'} | set(var_args['mut_fields'])
            ]))

        print('use_fields variable: ' + str(use_fields))
        print('use_cols variable: ' + str(use_cols))
        
    # Previously, this was read in from the Synapse repository due to HIPAA complience.
    # Loading data in locally with this experiment so don't need to access Synpase client.
    # "syn" variable in this case is the local .tsv file.
    
    print('syn variable: ' + str(syn))
    
    var_data = pd.read_csv(syn, sep='\t',
                           header=None, dtype='object', skiprows=1,
                           usecols=use_cols, names=use_fields)

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
        
        '''
        var_data.loc[(var_data.Form == 'frameshift_variant')
                   & (var_data.Variant_Class == 'insertion'),
                   'Form'] = 'Frame_Shift_Ins'
        var_data.loc[(var_data.Form == 'frameshift_variant')
                   & (var_data.Variant_Class == 'deletion'),
                   'Form'] = 'Frame_Shift_Del'
        '''
        
        # Edited this portion of the code because the annotation I added using Functotator
        # has values "INS" and "DEL" for deletion and insertion mutations.
        var_data.loc[(var_data.Form == 'frameshift_variant')
                   & (var_data.Variant_Class == 'INS'),
                   'Form'] = 'Frame_Shift_Ins'
        var_data.loc[(var_data.Form == 'frameshift_variant')
                   & (var_data.Variant_Class == 'DEL'),
                   'Form'] = 'Frame_Shift_Del'

    if 'ref_count' in use_fields:
        var_data['ref_count'] = var_data.total_reads - var_data.alt_count

    return var_data


def process_input_datasets(baml_dir, annot_dir, syn, **data_args):

    # Sample data file is stored locally and contains info on the samples/patients.
    # The main use of this file is to replace sample names with unidentifiable 
    # patient/sample identifiers.
    samp_data = pd.read_csv(os.path.join(baml_dir, 
                            "beataml_wv1to4_clinical_summary_worksheet_12_01_2021_w_dbgap_dict_survival_eln_rev.txt")
                            , sep='\t', encoding='latin-1')#encoding='utf-8')
                            # Added "encoding='utf-8'" because I was getting this error from trying to read this file
                            # in.. Error" "UnicodeDecodeError: 'utf-8' codec can't decode byte 0xca in position 45: 
                            # unexpected end of data"
    
    #print("samp_data variable = " + str(samp_data))
    
    expr = load_beat_expression(baml_dir)

    # Might need to update this to v43 or v44. The annot_dir variable is pointing 
    # to the following directory: '/home/groups/precepts/input-data/GENCODE/'.
    
    # Seems as though they used human genome "B37" as reference.. this is similar to GRCh37/hg19. 
    # When generating annotation with Funcotator tool, had to use the flag command:
    # --force-b37-to-hg19-reference-contig-conversion.. testing with:
    # gencode.v19.annotation.gtf.gz annotation file.
    annot_file = os.path.join(annot_dir, "gencode.v19.annotation.gtf.gz")
    
    if 'annot_fields' in data_args:
        annot_data = get_gencode(annot_file, data_args['annot_fields'])
    else:
        annot_data = get_gencode(annot_file)

    annot_dict = {at['gene_name']: {**{'Ens': ens}, **at}
                  for ens, at in annot_data.items()
                  if ens in set(expr.columns)}
    
    #print('annot_dict variable: ' + str(annot_dict))
    
    # TODO: incorporate supplemental mutation data, eg. laboratory-based
    # data for FLT3 ITDs found here:
    # https://www.nature.com/articles/s41586-018-0623-z
    variants = load_beat_variants(syn, **data_args)
    
    print('variants variable: ')
    print(variants)
    
    #variants['Sample'] = ['pid{}'.format(pid) for pid in variants.Sample.values]
    #samp_data['Sample'] = ['pid{}'.format(pid) for pid in samp_data.patientId]
    #use_samps = set(expr.index) & set(variants.Sample)
    use_samps = set(variants.Sample)
    
    print('use_samps variable: \n' + str(use_samps))

    ##### The "expr = expr.loc[use_samps, expr.columns.isin(annot_data)]" line of code is ##### 
    ##### throwing an error because there are some missing labels in the list..           #####
    # Full error:
    # KeyError: "Passing list-likes to .loc or [] with any missing labels is no longer supported. 
    # The following labels were missing: Index(['15-00180', '14-00171', '14-00212', '13-00654', 
    # '14-00536',\n       ...\n       '16-00709', '15-00805', '16-00886', '16-01123', '14-00449']
    # ,\n      dtype='object', length=256). See https://pandas.pydata.org/pandas-docs/stable/
    # user_guide/indexing.html#deprecate-loc-reindex-listlike"
    #expr = expr.loc[use_samps, expr.columns.isin(annot_data)]
    filtered_use_samps = set(expr.index) & use_samps
    expr = expr.loc[filtered_use_samps].filter(items=annot_data)
    expr_data = drop_duplicate_genes(expr.rename(
        columns={gn: annot_data[gn]['gene_name'] for gn in expr.columns}))

    print("variant variable exploration - why is it empty after filtering duplicates?")
    print("Original size:", len(variants))

    variants_sample_filtered = variants.loc[variants.Sample.isin(use_samps)]
    print("After sample filtering:", len(variants_sample_filtered))

    variants_gene_filtered = variants.loc[variants.Gene.isin(annot_dict.keys())]
    print("After gene filtering:", len(variants_gene_filtered))

    variants_no_duplicates = variants.loc[~variants.duplicated()]
    print("After removing duplicates:", len(variants_no_duplicates))
    
    # duplicates need to be filtered out here as they arise from two different
    # callers (varscan and mutect) being used to produce the mutation dataset
    variants = variants.loc[variants.Sample.isin(use_samps)
                            & variants.Gene.isin(annot_dict)]
    variants = variants.loc[~variants.duplicated()]
    
    print('##### Output of beatAML_waves_1_to_4.py - process_input_datasets() function #####')
    print('expr_data variable:')
    print(expr_data)
    print('^-- expr_data')
    print('variants variable:')
    print(variants)
    print('^-- variants')
    print('annot_dict variable:')
    print('Not pictured due to sheer volume.')
    #print(annot_dict)
    #print(annot_dict.keys())
    print('^-- annot_dict')

    return expr_data, variants, annot_dict


class BeatAmlCohort(BaseMutationCohort):

    def __init__(self,
                 baml_dir=None, annot_dir=None, syn=None, expr_data=None,
                 mut_data=None, annot_data=None, mut_levels=None,
                 mut_genes=None, collapse_txs=True,
                 leaf_annot=('ref_count', 'alt_count'),
                 cv_seed=None, test_prop=0, **coh_args):
        self.cohort = 'beatAMLwvs1to4'

        expr_data, mut_data, gene_annot = process_input_datasets(
            baml_dir, annot_dir, syn, collapse_txs, **coh_args)
 
        super().__init__(expr_data, mut_data, mut_levels, mut_genes,
                         gene_annot, leaf_annot, cv_seed, test_prop)

