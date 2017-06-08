
"""Loading and processing copy number alteration datasets.

This module contains functions for retrieving CNA data and processing it into
formats suitable for use in machine learning pipelines.

Author: Michal Grzadkowski <grzadkow@ohsu.edu>

"""

import pandas as pd
from ophion import Ophion
from . import DATA_PATH
from functools import reduce


def get_copies_firehose(cohort, gene_list):
    """Loads gene-level copy number alteration data from Broad Firehose.

    Args:
        cohort (str): The name of an cohort in Firehose.
        gene_list (list): The genes for which we want CNA data.

    Returns:
        copy_data (dict): A nested dictionary where the first level
                          corresponds to a gene and the second level
                          corresponds to sample. Note that this CNA data has
                          been thresholded and discretized to be integers in
                          the range [-2, 2].

    Examples:
        >>> copy_data1 = get_copies_firehose("BRCA", ["TP53", "PTEN"])
        >>> copy_data2 = get_copies_firehose("SKCM", ["PIK3CA"])

    """
    copy_table = pd.read_csv(DATA_PATH + '/copies/' + cohort
                             + '_all_thresholded.by_genes.txt.gz', sep='\t')

    # parse the TCGA barcodes to remove unnecessary suffixes
    copy_table.columns = [reduce(lambda x, y: x + '-' + y,
                                 samp.split('-', 4)[:4])
                          if 'TCGA' in samp else samp
                          for samp in copy_table.columns]

    # filter the copy-number data for the genes in the given list
    copy_data = {gene: copy_table.ix[copy_table.ix[:, 0] == gene, 3:]
                           .iloc[0, :].to_dict()
                 for gene in gene_list}

    return copy_data


def get_copies_bmeg(cohort, gene_list):
    """Loads gene-level copy number data from BMEG.

    Args:
        cohort (str): The name of an individualCohort vertex in BMEG.

    Returns:
        expr_data (pandas DataFrame of float), shape = [n_samps, n_feats]

    Examples:
        >>> copy_data = get_copies_bmeg('TCGA-OV')

    """
    oph = Ophion("http://bmeg.io")
    copy_list = {}

    for gene in gene_list:
        copy_list[gene] = {samp: 0 for samp in sample_gids}

        for i in oph.query().has("gid", "gene:" + gene)\
                .incoming("segmentInGene").mark("segment")\
                .outgoing("segmentOfSample")\
                .has("gid", oph.within(list(sample_gids))).mark("sample")\
                .select(["sample", "segment"]).execute():
            copy_list[gene][i["sample"]["name"]] = i["segment"]["value"]

    return pd.DataFrame(copy_list)
