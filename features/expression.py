
"""Loading and processing expression datasets.

This module contains functions for retrieving RNA-seq expression data
and processing it into formats suitable for use in machine learning pipelines.

Author: Michal Grzadkowski <grzadkow@ohsu.edu>

"""

from .utils import choose_bmeg_server

import numpy as np
import pandas as pd

import os
import glob

import tarfile
from io import BytesIO

import json
from ophion import Ophion


def log_norm_expr(expr):
    """Log-normalizes expression data.

    Puts a matrix of RNA-seq expression values into log-space after adding
    a constant derived from the smallest non-zero value.

    Args:
        expr (array of float), shape = [n_samples, n_features]

    Returns:
        norm_expr (array of float), shape = [n_samples, n_features]

    Examples:
        >>> norm_expr = log_norm_expr(np.array([[1.0, 0], [2.0, 8.0]]))
        >>> print(norm_expr)
                [[ 0.5849625 , -1.],
                 [ 1.32192809,  3.08746284]]

    """
    log_add = np.nanmin(expr[expr > 0]) * 0.5
    norm_expr = np.log2(expr + log_add)

    return norm_expr


def get_expr_bmeg(cohort):
    """Loads RNA-seq gene-level expression data from BMEG.

    Args:
        cohort (str): The name of an individualCohort vertex in BMEG.

    Returns:
        expr_data (pandas DataFrame of float), shape = [n_samps, n_feats]

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
    expr_mat = pd.DataFrame(expr_list).transpose().fillna(0.0)
    gene_set = expr_mat.columns
    expr_mat.index = [x[-1] for x in expr_mat.index.str.split(':')]
    expr_data = log_norm_expr(expr_mat.loc[:, gene_set])

    return expr_data


def get_expr_firehose(cohort, data_dir):
    """Loads RNA-seq gene-level expression data downloaded from Firehose.

    Args:
        cohort (str): The name of a TCGA cohort available in Broad Firehose.
        data_dir (str): The local directory where the Firehose data was
                        downloaded.

    Returns:
        expr_data (pandas DataFrame of float), shape = [n_samps, n_feats]

    Examples:
        >>> expr_data = get_expr_bmeg(
        >>>     'BRCA', '/home/users/grzadkow/compbio/input-data/firehose')
        >>> expr_data = get_expr_bmeg('SKCM', '../firehose')

    """
    expr_tar = tarfile.open(glob.glob(os.path.join(
        data_dir, "stddata__2016_01_28", cohort, "20160128",
        "*Merge_rnaseqv2_*_RSEM_genes_normalized_*.Level_3*.tar.gz"
        ))[0])

    expr_fl = expr_tar.extractfile(expr_tar.getmembers()[0])
    expr_data = pd.read_csv(BytesIO(expr_fl.read()),
                            sep='\t', skiprows=[1], index_col=0,
                            engine='python')

    expr_data = log_norm_expr(expr_data.transpose().fillna(0.0))
    expr_data.columns = [gn.split('|')[0] if isinstance(gn, str) else gn
                         for gn in expr_data.columns]

    return expr_data

