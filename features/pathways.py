
"""Loading and processing pathway datasets.

This module contains functions for retrieving gene pathway information
and processing it into formats suitable for use in machine learning pipelines.

Author: Michal Grzadkowski <grzadkow@ohsu.edu>

"""

import numpy as np
from itertools import groupby

from . import DATA_PATH
path_file = DATA_PATH + '/PathwayCommons9.All.hgnc.sif.gz'


def parse_sif(mut_genes):
    """Parses a SIF dataset to get the pathway neighbours of a set of genes.

    Args:
        mut_genes (list of str): Genes to get pathway neighbourhoods for.

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

    # loads the pathway data from file
    sif_dt = np.dtype([('UpGene', '|S16'),
                       ('Type', '|S32'),
                       ('DownGene', '|S16')])
    sif_data = np.genfromtxt(fname=path_file, dtype=sif_dt, delimiter='\t')

    # create neighbourhood dictionary with an entry for each given gene
    neighb = {gene: {'Up': None, 'Down': None} for gene in mut_genes}
    for gene in mut_genes:

        # filter pathway data for interactions involving each given gene,
        # separated by pathway interaction direction
        up_neighbs = sorted([(x['Type'].decode(), x['UpGene'].decode())
                             for x in sif_data
                             if x['DownGene'].decode() == gene],
                            key=lambda x: x[0])
        down_neighbs = sorted([(x['Type'].decode(), x['DownGene'].decode())
                               for x in sif_data
                               if x['UpGene'].decode() == gene],
                              key=lambda x: x[0])

        # further separate pathway data according to interaction type
        neighb[gene]['Up'] = {k: [x[1] for x in v] for k, v in
                              groupby(up_neighbs, lambda x: x[0])}
        neighb[gene]['Down'] = {k: [x[1] for x in v] for k, v in
                                groupby(down_neighbs, lambda x: x[0])}

    return neighb
