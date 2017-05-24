
"""
HetMan (Heterogeneity Manifold)
Classification of mutation sub-types using expression data.
This file contains functions for loading and processing expression datasets.
"""

# Author: Michal Grzadkowski <grzadkow@ohsu.edu>

import numpy as np
import pandas as pd

import json
from ophion import Ophion


def log_norm_expr(expr):
    """Log-normalizes expression data."""
    log_add = np.nanmin(expr[expr > 0].values) * 0.5
    return np.log2(expr + log_add)


def get_expr_bmeg(cohort):
    """Loads RNA-seq expression data from BMEG."""

    oph = Ophion("http://bmeg.io")
    expr_list = {}

    # TODO: filter on gene chromosome when BMEG is updated
    expr_query = ('oph.query().has("gid", "individualCohort:" + cohort)'
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
    for i in eval(expr_query).execute():
        dt = json.loads(i)
        if ('expression' in dt and 'properties' in dt['expression']
                and 'serializedExpressions' in dt['expression']['properties']):
            expr_list[dt['sample']['gid']] = json.loads(
                dt['expression']['properties']['serializedExpressions'])

    # creates a sample x expression matrix and normalizes it
    expr_mat = pd.DataFrame(expr_list).transpose().fillna(0.0)
    gene_set = expr_mat.columns
    expr_data = log_norm_expr(expr_mat.loc[:, gene_set])

    return expr_data

