
"""Loading and processing expression datasets.

This module contains functions for retrieving RNA-seq expression data
and processing it into formats suitable for use in machine learning pipelines.

Author: Michal Grzadkowski <grzadkow@ohsu.edu>

"""

from HetMan.features.data.utils import choose_bmeg_server
import numpy as np
import pandas as pd

import os
import glob

import gzip
import tarfile
from io import BytesIO

import json
from ophion import Ophion


def get_expr_bmeg(cohort):
    """Loads RNA-seq gene-level expression data from BMEG.

    Args:
        cohort (str): The name of an individualCohort vertex in BMEG.

    Returns:
        expr_data (:obj:`pd.DataFrame`, shape = [n_samps, n_genes])

    Examples:
        >>> expr_data = get_expr_bmeg('TCGA-BRCA')
        >>> expr_data = get_expr_bmeg('TCGA-PCPG')

    """
    oph = Ophion(choose_bmeg_server(verbose=True))
    expr_list = {}

    # TODO: filter on gene chromosome when BMEG is updated
    expr_query = ('oph.query().has("gid", "project:" + cohort)'
                  '.outgoing("hasMember").incoming("biosampleOfIndividual")'
                  '.mark("sample").incoming("expressionForSample")'
                  '.mark("expression").select(["sample", "expression"])')
    samp_count = eval(expr_query).count().execute()[0]

    # ensures BMEG is running
    if not samp_count.isnumeric():
        raise IOError("BMEG could not process query, returned error:\n"
                      + samp_count)

    # ensures the query returns data
    if samp_count == '0':
        raise ValueError("No samples found in BMEG for cohort "
                         + cohort + " !")

    # parses expression data and loads it into a list
    for qr in eval(expr_query).execute():
        dt = json.loads(qr)
        if ('expression' in dt and 'properties' in dt['expression']
                and 'serializedExpressions'
                in dt['expression']['properties']):
            expr_list[dt['sample']['gid']] = json.loads(
                dt['expression']['properties']['serializedExpressions'])

    # creates a sample x expression matrix and normalizes it
    expr_mat = pd.DataFrame(expr_list).transpose()
    expr_mat.index = [x[-1] for x in expr_mat.index.str.split(':')]

    return expr_mat


def get_expr_firehose(cohort, data_dir):
    """Loads RNA-seq gene-level expression data downloaded from Firehose.

    Args:
        cohort (str): The name of a TCGA cohort available in Broad Firehose.
        data_dir (str): The local directory where the Firehose data was
                        downloaded.

    Returns:
        expr_data (:obj:`pd.DataFrame`, shape = [n_samps, n_genes])

    Examples:
        >>> expr_data = get_expr_bmeg(
        >>>     'BRCA', '/home/users/grzadkow/compbio/input-data/firehose')
        >>> expr_data = get_expr_bmeg('SKCM', '../firehose')

    """

    # finds the tarballs containing expression data for the given cohort in
    # the given data directory
    expr_tars = glob.glob(os.path.join(
        data_dir, "stddata__2016_01_28", cohort, "20160128",
        ("*.Merge_rnaseqv2__illuminahiseq_rnaseqv2__unc_edu__Level_3__"
         "RSEM_genes_normalized__data.Level_3.2016012800.0.0.tar.gz")
        ))

    # ensures only one tarball matches the file name pattern
    if len(expr_tars) > 1:
        raise IOError("Multiple normalized gene expression tarballs found "
                      "for cohort {} in directory {} !".format(
                          cohort, data_dir))

    elif len(expr_tars) == 0:
        raise IOError("No normalized gene expression tarballs found "
                      "for cohort {} in directory {} !".format(
                          cohort, data_dir))

    # opens the tarball and locates the expression data within it
    expr_tar = tarfile.open(expr_tars[0])
    expr_indx = [i for i, memb in enumerate(expr_tar.getmembers())
                 if 'data.txt' in memb.get_info()['name']]
  
    # ensures only one file in the tarball contains expression data
    if len(expr_indx) == 0:
        raise IOError("No expression files found in the tarball!")
    elif len(expr_indx) > 1:
        raise IOError("Multiple expression files found in the tarball!")

    expr_fl = expr_tar.extractfile(expr_tar.getmembers()[expr_indx[0]])
    expr_data = pd.read_csv(BytesIO(expr_fl.read()),
                            sep='\t', skiprows=[1], index_col=0,
                            engine='python').transpose()

    expr_data.columns = [gn.split('|')[0] if isinstance(gn, str) else gn
                         for gn in expr_data.columns]
    expr_data = expr_data.iloc[:, expr_data.columns != '?']
    expr_data.index = ["-".join(x[:4])
                       for x in expr_data.index.str.split('-')]

    return expr_data


def get_expr_cptac(syn, cohort):
    """Get the CPTAC RNAseq data used in the DREAM proteogenomics challenge.

    Args:
        syn (synapseclient.Synapse): A logged-into Synapse instance.
        cohort (str): A TCGA cohort included in the challenge.

    Examples:
        >>> import synapseclient
        >>> syn = synapseclient.Synapse()
        >>>
        >>> syn.cache.cache_root_dir = (
        >>>     '/home/exacloud/lustre1/'
        >>>     'share_your_data_here/precepts/synapse'
        >>>     )
        >>> syn.login()
        >>>
        >>> get_expr_cptac(syn, "BRCA")
        >>> get_expr_cptac(syn, "OV")

    """

    syn_ids = {'BRCA': '11328694',
               'OV': '10535396'}

    expr_data = pd.read_csv(syn.get("syn{}".format(syn_ids[cohort])).path,
                            sep='\t', index_col=0).transpose()

    return expr_data


def get_expr_icgc(cohort, data_dir):
    """Loads RNA-seq gene-level expression data downloaded from ICGC.

    Args:
        cohort (str): The name of an ICGC downloaded locally.
        data_dir (str): The local directory where the ICGC data has
                        been downloaded.

    Returns:
        expr_data (:obj:`pd.DataFrame`, shape = [n_samps, n_genes])

    Examples:
        >>> expr_data = get_expr_icgc(
        >>>     'PACA-AU',
        >>>     '/home/exacloud/lustre1/share_your_data_here/precepts/ICGC'
        >>>     )
        >>> expr_data = get_expr_icgc('LIRI-JP', '../ICGC-cohorts')

    """

    expr_file = os.path.join(data_dir, cohort, 'exp_seq.tsv.gz')
    expr_df = pd.read_csv(expr_file, sep='\t')
    expr_data = expr_df.pivot(index='icgc_sample_id', columns='gene_id',
                              values='normalized_read_count')

    return expr_data


def get_expr_toil(cohort, data_dir, collapse_txs=False):
    expr_file = os.path.join(data_dir, '{}.txt.gz'.format(cohort))
    expr_data = pd.read_csv(expr_file, sep='\t', index_col=0).transpose()
    expr_data.columns.name = None

    # loads mapping between transcripts and the genes they are a part of
    tx_file = ('/home/exacloud/lustre1/CompBio/mgrzad/input-data'
               '/toil/gencode.v23.annotation.transcript.probemap.gz')
    tx_annot = pd.read_csv(tx_file, sep='\t', index_col=0)

    expr_data.columns = pd.MultiIndex.from_arrays(
        [tx_annot.loc[expr_data.columns, 'gene'], expr_data.columns],
        names=['Gene', 'Transcript']
        )
    
    if collapse_txs:
        expr_data = np.log2(
            expr_data.rpow(2).subtract(0.001).groupby(
                level=['Gene'], axis=1).sum().add(0.001)
            )

    else:
        expr_data.sort_index(axis=1, level=['Gene'], inplace=True)

    return expr_data

