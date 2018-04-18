
"""Loading and processing pathway datasets.

This module contains functions for retrieving gene pathway information
and processing it into formats suitable for use in machine learning pipelines.

Author: Michal Grzadkowski <grzadkow@ohsu.edu>

"""

import pandas as pd
from itertools import groupby

from HetMan.features import DATA_PATH
path_file = DATA_PATH + '/PathwayCommons9.All.hgnc.sif.gz'


def load_sif():
    return pd.read_csv(path_file,
                       names=['UpGene', 'Type', 'DownGene'],
                       sep='\t', header=None)


def get_gene_neighbourhood(genes):
    """Parses a SIF dataset to get the pathway neighbours of a set of genes.

    Args:
        genes (list of str): Genes to get pathway neighbourhoods for.

    Returns:
        neighb (dict): Neigbourhood info for each gene.
            The neighbourhood is represented as a nested dictionary with three
            levels, with the first level corresponding to genes, the second
            level corresponding to pathway interaction direction (upstream or
            downstream), and the third level corresponding to interaction
            type, eg. 'controls-phosphorylation-of', 'interacts-with', or
            'controls-expression-of'.

    Examples:
        >>> parse_sif(['TP53'])
        >>> parse_sif(['PIK3CA', 'RB1', 'ACT1'])

    """
    neighb = {gene: {'Up': {}, 'Down': {}} for gene in genes}
    sif_data = load_sif()

    # sorts upstream interactions according to downstream mutated gene
    # and interaction type
    up_data = sif_data.loc[sif_data['DownGene'].isin(genes), :]
    for (gn, tp), dt in up_data.groupby(['DownGene', 'Type']):
        neighb[gn]['Up'][tp] = set(dt['UpGene'])

    # sorts downstream interactions according to upstream mutated gene
    # and interaction type
    down_data = sif_data.loc[sif_data['UpGene'].isin(genes), :]
    for (gn, tp), dt in down_data.groupby(['UpGene', 'Type']):
        neighb[gn]['Down'][tp] = set(dt['DownGene'])

    return neighb


def get_type_networks(intx_types=None, genes=None):
    """Parses a SIF dataset to get the interactions of the given type(s).

    """
    intx_data = load_sif()

    if intx_types is not None:
        intx_data = intx_data.loc[intx_data['Type'].isin(intx_types), :]

    if genes is not None:
        intx_data = intx_data.loc[intx_data['UpGene'].isin(genes)
                                  & intx_data['DownGene'].isin(genes), :]
    
    neighb = {
        tp: set(tuple(x) for x in
                dt.loc[:, ['UpGene', 'DownGene']].values.tolist())
        for tp, dt in intx_data.groupby(['Type'])
        }

    return neighb

