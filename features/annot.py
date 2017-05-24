
"""
HetMan (Heterogeneity Manifold)
Classification of mutation sub-types using expression data.
This file contains functions for loading gene annotation data and parsing
it into useful data structures.
"""

# Author: Michal Grzadkowski <grzadkow@ohsu.edu>

import numpy as np
import pandas as pd

from . import DATA_PATH

from re import sub as gsub


def get_gencode():
    """Gets annotation data for protein-coding genes on non-sex
       chromosomes from a Gencode file.

    Returns
    -------
    annot : dict
        Dictionary with keys corresponding to Ensembl gene IDs and values
        consisting of dicts with annotation fields.
    """
    annot = pd.read_csv(DATA_PATH + "gencode.v22.annotation.gtf.gz",
                        usecols=[0, 2, 3, 4, 8],
                        names=['Chr', 'Type', 'Start', 'End', 'Info'],
                        sep='\t', header=None, comment='#')

    # filter out annotation records that aren't
    # protein-coding genes on non-sex chromosomes
    chroms_use = ['chr' + str(i+1) for i in range(22)]
    annot = annot.loc[annot['Type'] == 'gene', ]
    chr_indx = np.array([chrom in chroms_use for chrom in annot['Chr']])
    annot = annot.loc[chr_indx, ]

    # parse the info field to get each gene's annotation data
    gn_annot = {gsub('\.[0-9]+', '', z['gene_id']).replace('"', ''): z
                for z in [dict([['chr', an[0]]]
                               + [['Start', an[2]]] + [['End', an[3]]] +
                               [y for y in [x.split(' ')
                                            for x in an[4].split('; ')]
                                if len(y) == 2])
                          for an in annot.values]
                if z['gene_type'] == '"protein_coding"'}

    for g in gn_annot:
        gn_annot[g]['gene_name'] = gn_annot[g]['gene_name'].replace('"', '')

    return gn_annot

