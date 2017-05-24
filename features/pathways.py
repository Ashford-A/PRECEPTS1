
"""
HetMan (Heterogeneity Manifold)
Classification of mutation sub-types using expression data.
This file contains functions for loading pathway interaction data and parsing
it into useful data structures.
"""

# Author: Michal Grzadkowski <grzadkow@ohsu.edu>

from . import DATA_PATH
path_file = DATA_PATH + '/PathwayCommons9.All.hgnc.sif.gz'

import numpy as np
from itertools import groupby


def parse_sif(mut_genes):
    """Parses a sif dataset to get the pathway neighbours of a given gene."""

    sif_dt = np.dtype([('UpGene', '|S16'),
                       ('Type', '|S32'),
                       ('DownGene', '|S16')])
    sif_data = np.genfromtxt(fname=path_file, dtype=sif_dt, delimiter='\t')

    neighb = {}
    for gene in mut_genes:
        neighb[gene] = {}

        # TODO: add 'other' type interactions
        up_neighbs = sorted([(x['Type'].decode(), x['UpGene'].decode())
                             for x in sif_data
                             if x['DownGene'].decode() == gene],
                            key=lambda x: x[0])
        down_neighbs = sorted([(x['Type'].decode(), x['DownGene'].decode())
                               for x in sif_data
                               if x['UpGene'].decode() == gene],
                              key=lambda x: x[0])

        # parses according to interaction type
        neighb[gene]['Up'] = {k: [x[1] for x in v] for k, v in
                              groupby(up_neighbs, lambda x: x[0])}
        neighb[gene]['Down'] = {k: [x[1] for x in v] for k, v in
                                groupby(down_neighbs, lambda x: x[0])}

    return neighb

