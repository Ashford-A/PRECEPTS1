"""Utilities for loading and consolidating cohort data from any source."""

from ...experiments.utilities.data_dirs import (
    firehose_dir, syn_root, metabric_dir, baml_dir, ccle_dir,
    gencode_dir, subtype_file, expr_sources
    )

from .beatAML import process_input_datasets as process_baml_datasets

# Added by Andrew to process the BeatAML waves 1-4 input datasets:
from .beatAML_waves_1_to_4 import process_input_datasets as process_baml_waves_1_to_4_datasets

from .tcga import tcga_subtypes
from .tcga import process_input_datasets as process_tcga_datasets
from .tcga import choose_subtypes as choose_tcga_subtypes
from .tcga import parse_subtypes as parse_tcga_subtypes

from .metabric import process_input_datasets as process_metabric_datasets
from .metabric import choose_subtypes as choose_metabric_subtypes
from .metabric import list_subtypes as list_metabric_subtypes
from .metabric import load_metabric_samps
from .ccle import process_input_datasets as process_ccle_datasets

from dryadic.features.data.vep import process_variants
from dryadic.features.cohorts.mut import BaseMutationCohort

import os
import synapseclient
import pandas as pd
import dill as pickle


def get_input_datasets(cohort, expr_source, mut_fields=None):
    """Loads normalized (but not cleaned) data for an instance of a cohort.

    Arguments:
        cohort (str): A cancer cohort, eg. 'METABRIC', 'HNSC_HPV-', etc.
        expr_source (str): A place or method such as 'Firehose' or
                           'microarray' that describes how the expression
                           calls were made or stored.
        mut_fields (:obj: `iterable` of :obj: `str`, optional)
            Which mutation annotation fields to load for the cohort.

    Returns:
        data_dict (dict): The data for the cohort, which includes expression,
                          variant calls, gene annotations, copy number
                          alteration calls where available, and the genome
                          assembly used to create the above.

    """
    
    data_dict = {data_k: None
                 for data_k in ('expr', 'vars', 'copy', 'annot', 'assembly')}
    
    
    # Need this to load in Synapse repo files! Currently commented out because 
    # I'm currently not accessing the Synapse repo.
    
    
    # instantiate Synapse client, find where locally saved credentials are
    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = syn_root
    
    
    # if loading the beatAML cohort, we pull the datasets from Synapse
    if cohort == 'beatAML':
        syn.login()
        data_dict['assembly'] = 'GRCh37'

        if expr_source != 'toil__gns':
            raise ValueError("Only gene-level Kallisto calls are available "
                             "for the beatAML cohort!")

        data_dict.update({
            data_k: baml_data
            for data_k, baml_data in zip(
                ('expr', 'vars', 'annot'),
                process_baml_datasets(baml_dir, gencode_dir, syn,
                                      annot_fields=['transcript'],
                                      mut_fields=mut_fields)
                )
            })
        
    ########################################################################
    # Added by Andrew on 9/12/2023 to account for the case when we want to #
    # use beatAML waves 1-4 data to train the models..                     #
    ########################################################################
    
    elif cohort == 'beatAMLwvs1to4':
        # The commented out code is the code above to load in the original 
        # beatAML data waves 1 and 2 for reference. We don't need to use 
        # Synapse client in this case..
        '''
        syn.login()
        data_dict['assembly'] = 'GRCh37'

        if expr_source != 'toil__gns':
            raise ValueError("Only gene-level Kallisto calls are available "
                             "for the beatAML cohort!")

        data_dict.update({
            data_k: baml_data
            for data_k, baml_data in zip(
                ('expr', 'vars', 'annot'),
                process_baml_datasets(baml_dir, gencode_dir, syn,
                                      annot_fields=['transcript'],
                                      mut_fields=mut_fields)
                )
            })
        '''
        
        data_dict['assembly'] = 'GRCh37' # Might need to change this to GRCh38..
         
        '''
        Simplifying the data read in in this script by taking out the use of environmental 
        variables.. Specify the data locations below.
        '''
        # Specify RNA-seq tpm-normalized expression data directory and filename variables below.
        # Eventually maybe add this to the script "run_analysis.sh" to specify it in one line.
        ########################################################################################
        baml_dir = '/home/groups/precepts/ashforda/single-cell_precepts/custom_precepts_inputs/BeatAML_waves_1-4_training/'
        ########################################################################################
        
        # Specify the location of the "syn" file. For BeatAML waves 1-4 it's stored on Exacloud
        # in the following location:
        syn = baml_dir + 'beataml_wes_wv1to4_mutations_v4_exon_mut_loc_and_variant_class_added.tsv'
        
        data_dict.update({
            data_k: baml_data
            for data_k, baml_data in zip(
                ('expr', 'vars', 'annot'),
                process_baml_waves_1_to_4_datasets(baml_dir, gencode_dir, syn,
                                      annot_fields=['transcript'],
                                      mut_fields=mut_fields)
                )
            })
    
        #print(str(data_dict) + ' -> data_dict updated.')
        #print('data_dict keys: ' + str(data_dict.keys()))
        #print('data_dict at vars: ' + str(data_dict['vars']))
    
    ########################################################################

    
    # if loading the METABRIC cohort, we pull the datasets from where they
    # were locally downloaded to from cBioPortal
    elif cohort.split('_')[0] == 'METABRIC':
        data_dict['assembly'] = 'GRCh37'
              
        if expr_source != 'microarray':
            raise ValueError("Only Illumina microarray mRNA calls are "
                             "available for the METABRIC cohort!")

        if '_' in cohort:
            use_types = cohort.split('_')[1]
        else:
            use_types = None

        data_dict.update({
            data_k: mtbc_data
            for data_k, mtbc_data in zip(
                ('expr', 'vars', 'copy', 'annot'),
                process_metabric_datasets(
                    metabric_dir, gencode_dir, use_types,
                    annot_fields=['transcript'], mut_fields=mut_fields
                    )
                )
            })

    # if we are loading the CCLE cohort, we also pull the datasets from where
    # they were locally downloaded
    elif cohort.split('_')[0] == 'CCLE':
        data_dict['assembly'] = 'GRCh37'

        data_dict.update({
            data_k: ccle_data
            for data_k, ccle_data in zip(
                ('expr', 'vars', 'copy', 'annot'),
                process_ccle_datasets(ccle_dir, gencode_dir, expr_source)
                )
            })

    # ...otherwise, assume we are dealing with a TCGA dataset; data can come
    # from Firehose, or other sources such as locally-stored Kallisto calls
    else:
        syn.login()
        data_dict['assembly'] = 'GRCh37'

        source_info = expr_source.split('__')
        source_base = source_info[0]
        collapse_txs = not (len(source_info) > 1 and source_info[1] == 'txs')

        data_dict.update({
            data_k: tcga_data
            for data_k, tcga_data in zip(
                ('expr', 'vars', 'copy', 'annot'),
                process_tcga_datasets(
                    cohort, expr_source=source_base,
                    var_source='mc3', copy_source='Firehose',
                    expr_dir=expr_sources[source_base],
                    copy_dir=expr_sources['Firehose'], annot_dir=gencode_dir,
                    type_file=subtype_file, collapse_txs=collapse_txs,
                    annot_fields=['transcript'], syn=syn,
                    mut_fields=mut_fields
                    )
                )
            })

    return data_dict


def get_cohort_data(cohort, expr_source, mut_lvls, vep_cache_dir, out_path,
                    use_genes=None, use_copies=True, leaf_annot=None):
    """Creates a mutation cohort object using expression and mutation data.

    This function uses :func:`get_input_datasets` to get the datasets
    associated with a given cohort, and then consolidates that data into a
    BaseMutationCohort instance that can be used for both mutation subgrouping
    enumeration and classification experiments.

    A crucial intermediate step in this process is running the VEP mutation
    annotation tool on the given mutation data to produce variant calls that
    are standardized across all cohorts.

    The <expr_source> + <cohort> combinations currently supported are:
        microarray  METABRIC
        Firehose    any TGCA cohort available through GDAC
                        e.g. BLCA, BRCA, HNSC, LUAD, LUSC
        toil__gns   beatAML, any TGCA cohort available through GDAC
        toil__txs   any TCGA cohort available through GDAC
        <>          CCLE (any expr source can be given)

    Args:
        cohort (str): A cancer cohort, eg. 'METABRIC', 'HNSC_HPV-', etc.
        expr_source (str): A place or method such as 'Firehose' or
                           'microarray' that describes how the expression
                           calls were made or stored.

        mut_lvls (:obj: `iterable` of :obj: `str`, optional)
            Which combination(s) of mutation annotation fields to use for
            constructing the hierarchy of mutations for this cohort.
                e.g. ['Consequence', 'Exon']
                     ['Pfam-domain', 'Position', 'HGVSp']
            This can either be a list of fields for one combination, or a list
            of such lists to use any number of combinations.

        vep_cache_dir (str): Where VEP is storing genome assembly datasets.
        out_path (str): Where to store intermediate output files.

        use_genes (:obj: `iterable` of :obj: `str`, optional)
            If given, only mutations from these genes will be included.
            Default is to use all genes with mutations in the cohort.
        use_copies (boolean, optional)
            Whether to include copy number alterations in the mutation data.
        leaf_annot (:obj: `iterable` of :obj: `str`, optional)
            Which mutation properties to use for annotating the leaf nodes
            of each mutation tree constructed for the cohort. Default is to
            not use leaf annotations.

    Returns:
        cdata (BaseMutationCohort)

    """

    # these mutation annotation fields need to be pulled from the cohort's
    # mutation data in order to produce input for VEP
    mut_fields = ['Sample', 'Gene', 'Chr', 'Start', 'End',
                  'RefAllele', 'TumorAllele']
    if leaf_annot is not None:
        mut_fields += leaf_annot

    print(cohort)
    print(expr_source)
    
    data_dict = get_input_datasets(cohort, expr_source, mut_fields=mut_fields)

    # TODO: how to handle this special case
    if cohort == 'CCLE':
        return BaseMutationCohort(data_dict['expr'], data_dict['vars'],
                                  [('Gene', 'Form')], data_dict['copy'],
                                  data_dict['annot'], leaf_annot=None)
    
    elif leaf_annot is not None:
        use_vars = data_dict['vars'].loc[
            data_dict['vars'].Gene.isin(use_genes)]
        use_copy = data_dict['copy'].loc[
            data_dict['copy'].Gene.isin(use_genes)]
        
        print('use_vars variable: ' + str(use_vars))
        print('use_copy variable: ' + str(use_copy))
        
        if 'ref_count' in leaf_annot:
            use_vars.loc[:, 'ref_count'] = use_vars.ref_count.astype(int)
        if 'depth' in leaf_annot:
            use_vars.loc[:, 'depth'] = use_vars.depth.astype(int)
        if 'alt_count' in leaf_annot:
            use_vars.loc[:, 'alt_count'] = use_vars.alt_count.astype(int)

        return BaseMutationCohort(data_dict['expr'], use_vars,
                                  mut_lvls, use_copy,
                                  data_dict['annot'], leaf_annot=leaf_annot)

    print('Chr to add to var_df: ')
    print(str(data_dict['vars'].Chr.astype('int')))
    
    # format the mutation calls into a format compatible with VEP
    var_df = pd.DataFrame({'Chr': data_dict['vars'].Chr.astype('int'),
                           'Start': data_dict['vars'].Start.astype('int'),
                           'End': data_dict['vars'].End.astype('int'),
                           'RefAllele': data_dict['vars'].RefAllele,
                           'VarAllele': data_dict['vars'].TumorAllele,
                           'Sample': data_dict['vars'].Sample})

    if isinstance(mut_lvls[0], str):
        var_fields = ['Gene', 'Canonical', 'Location', 'VarAllele']
        cdata_lvls = [mut_lvls]

        for lvl in mut_lvls:
            if '-domain' in lvl and 'Domains' not in var_fields:
                var_fields += ['Domains']
            elif lvl not in {'Gene', 'Scale', 'Copy'}:
                var_fields += [lvl]

    elif isinstance(mut_lvls[0], tuple):
        var_fields = {'Gene', 'Canonical', 'Location', 'VarAllele'}
        cdata_lvls = list(mut_lvls)

        for lvl_list in mut_lvls:
            for lvl in lvl_list:
                if '-domain' in lvl and 'Domains' not in var_fields:
                    var_fields |= {'Domains'}
                elif lvl not in {'Gene', 'Scale', 'Copy'}:
                    var_fields |= {lvl}

    else:
        raise TypeError(
            "Unrecognized <mut_lvls> argument: `{}`!".format(mut_lvls))

    ##### Print statements added to check for what's causing an error #####
    #print("data_dict: " + str(data_dict))
    print("var_df: " + str(var_df)) # Check where var_df is created to see if empyu output.. trace back.
    #### Where is vars.txt being written in the code?
    
    ##### Added by Andrew to check for an error #####
    print('Chr: ' + str(data_dict['vars'].Chr.astype('int')))
    print('RefAllele: ' + str(data_dict['vars'].RefAllele))
    
    #print('data_dict keys: ' + str(data_dict.keys))
    #print('data_dict at vars keys: ' + str(data_dict['vars'].keys))
        
    # run the VEP command line wrapper to obtain a standardized
    # set of point mutation calls
    
    print('var_fields variable: ' + str(var_fields))
    print('vep_cache_dir variable: ' + str(vep_cache_dir))
    print('out_path variable: ' + str(out_path))
    print('assembly in data dict: ' + str(data_dict['assembly']))
    
    variants = process_variants(
        var_df, out_fields=var_fields, cache_dir=vep_cache_dir,
        temp_dir=out_path, assembly=data_dict['assembly'],
        distance=0, consequence_choose='pick', forks=4, update_cache=False
        )

    # handle case where we don't need CNA data, where there is no CNA data
    # available for the cohort, and where there is CNA data respectively
    if not use_copies:
        copies = None
    elif data_dict['copy'] is None:
        copies = pd.DataFrame(columns=['Gene', 'Copy'])
    else:
        copies = data_dict['copy']

    # remove mutation calls not assigned to a canonical transcript by VEP as
    # well as those not associated with genes linked to cancer processes
    
    ##### COMMENTED OUT 9/27/2023 by Andrew #####
    #variants = variants.loc[variants.CANONICAL == 'YES'] #...my spreadsheet don't gots this..
    
    if use_genes:
        variants = variants.loc[variants.Gene.isin(use_genes)]

        if copies is not None:
            copies = copies.loc[copies.Gene.isin(use_genes)]

    assert not variants.duplicated().any(), (
        "Variant data contains {} duplicate entries!".format(
            variants.duplicated().sum())
        )

    return BaseMutationCohort(data_dict['expr'], variants, cdata_lvls, copies,
                              data_dict['annot'], leaf_annot=None)


def load_cohort(cohort, expr_source, mut_lvls, vep_cache_dir, use_path=None,
                temp_path=None, use_genes=None, leaf_annot=None):
    """Load a saved cohort object from file; create a new one if necessary."""

    if isinstance(mut_lvls[0], str):
        mut_lvls = [mut_lvls]

    if use_path is not None and os.path.exists(use_path):
        try:
            with open(use_path, 'rb') as f:
                cdata = pickle.load(f)

        except IOError:
            cdata = get_cohort_data(cohort, expr_source, mut_lvls,
                                    vep_cache_dir, temp_path, use_genes,
                                    leaf_annot=leaf_annot)

    else:
        cdata = get_cohort_data(cohort, expr_source, mut_lvls,
                                vep_cache_dir, temp_path, use_genes,
                                leaf_annot=leaf_annot)

    if cohort != 'CCLE' and not all(any(mtree_lvls[-(len(lvls)):] == lvls
                                        for mtree_lvls in cdata.mtrees)
                                    for lvls in mut_lvls):
        cdata.merge(get_cohort_data(cohort, expr_source, mut_lvls,
                                    vep_cache_dir, temp_path, use_genes,
                                    leaf_annot=leaf_annot))

    return cdata


def get_cohort_subtypes(coh):
    """Gets the cohort samples associated with known molecular subtypes."""

    if coh == 'METABRIC':
        metabric_samps = load_metabric_samps(metabric_dir)
        subt_dict = {subt: choose_metabric_subtypes(metabric_samps, subt)
                     for subt in ('LumA', 'luminal', 'nonbasal')}

    elif coh in tcga_subtypes:
        subt_dict = {
            subt: choose_tcga_subtypes(
                parse_tcga_subtypes("_{}".format(subt)), coh, subtype_file)
            for subt in tcga_subtypes[coh]
            }

    else:
        subt_dict = dict()

    return subt_dict


# TODO: consolidate this with the above function?
def list_cohort_subtypes(coh):
    if coh == 'METABRIC':
        type_dict = list_metabric_subtypes(load_metabric_samps(metabric_dir))

    elif (coh == 'beatAML') or (coh == 'beatAMLwvs1to4'):
        type_dict = {}

    else:
        tcga_types = pd.read_csv(subtype_file,
                                 sep='\t', index_col=0, comment='#')
        tcga_types = tcga_types.loc[tcga_types.DISEASE == coh]
        type_dict = tcga_types.groupby('SUBTYPE').groups

    return type_dict

