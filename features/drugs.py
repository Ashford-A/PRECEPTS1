
"""Loading and processing expression datasets.

This module contains functions for retrieving RNA-seq expression data
and processing it into formats suitable for use in machine learning pipelines.

Author: Michal Grzadkowski <grzadkow@ohsu.edu>

"""

from . import DATA_PATH

import pandas as pd
from scipy import stats

from ophion import Ophion
from fuzzywuzzy import process
import json


def exp_norm(expr):
    out_expr = expr.apply(
        lambda x: stats.expon.ppf((x.rank()-1) / len(x)), 1)
    return out_expr.fillna(0.0)


def get_expr_ioria():
    """Get the cell-line expression data used in the Ioria landscape study."""

    cell_expr = pd.read_csv(
        DATA_PATH + 'drugs/ioria/Cell_line_RMA_proc_basalExp.txt.gz',
        sep='\t', comment='#'
        )

    cell_expr = cell_expr.ix[~pd.isnull(cell_expr['GENE_SYMBOLS']), :]
    cell_expr.index = cell_expr['GENE_SYMBOLS']
    cell_expr = cell_expr.ix[:, 2:].transpose()

    return cell_expr


def get_drug_ioria(drug_list):
    """Get drug response data as collected by the Ioria landscape study.

    Args:
        drug_list (list of str): Which drugs to get responses for. Drug names
                                 can be approximate, in which case the best
                                 matching drug name available will be used.

    Returns:
        drug_resp (pandas DataFrame), shape = [n_samples, n_drugs]

    Examples:
        >>> drug_resp1 = get_drug_ioria(['AZ628'])
        >>> drug_resp2 = get_drug_ioria(['Trametinib', 'Nutlin-3a'])
        >>> drug_resp3 = get_drug_ioria(['Olaparxx', 'RD119'])
        >>> print(drug_resp3.columns)
            ['Olaparib', 'RDEA119']

    """
    drug_annot = pd.read_csv(DATA_PATH + 'drugs/ioria/drug_annot.txt.gz',
                             sep='\t', comment='#')
    drug_resp = pd.read_csv(DATA_PATH + 'drugs/ioria/drug-auc.txt.gz',
                            sep='\t', comment='#', index_col=0)

    # gets closest matching drug names available, retrieves corresponding drug
    # IDs used in the dataset
    drug_match = [(process.extractOne(drug, drug_annot['Name']),
                   process.extractOne(drug, drug_annot['Synonyms']))
                  for drug in drug_list]
    match_indx = [mtch[0] if mtch[0][1] > mtch[1][1] else mtch[1]
                  for mtch in drug_match]
    drug_lbl = ['X' + str(drug_annot['Identifier'][mtch[2]])
                for mtch in match_indx]

    # filter out drugs we don't need and replace approximate
    # names with matches
    drug_resp = drug_resp.loc[:, drug_lbl]
    drug_resp.columns = [mtch[0] if mtch[1] == 100
                         else drg + '__' + mtch[0]
                         for drg, mtch in zip(drug_list, match_indx)]

    return drug_resp


def get_drug_bmeg(cohort, drug_list):
    """Get cell-line drug response data from BMEG.

    Args:
        cohort (str): The name of an individualCohort vertex in BMEG.
        drug_list (list of str): Which drugs to get responses for.

    Returns:

    Examples:

    """
    oph = Ophion("http://bmeg.io")
    drug_data = {drug: None for drug in drug_list}

    return drug_expr

def get_drug_annot_bmeg(drug_list):
    """Get annotation data for specified drugs (i.e. target).

    Args:
        drug_list (list of str): List of drug names to be annotated

    """
    # how to handle drug name mismatches?

    oph = Ophion("http://bmeg.io")

    # some ophion query to fetch targets for drugs in drug_list

    pass
