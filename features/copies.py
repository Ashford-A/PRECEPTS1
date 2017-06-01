
"""
HetMan (Heterogeneity Manifold)
Classification of mutation sub-types using expression data.
This file contains classes for representing and storing
copy number alterations.
"""

# Author: Michal Grzadkowski <grzadkow@ohsu.edu>

import pandas as pd
from ophion import Ophion


def get_copies_bmeg(sample_gids, gene_list):
    """Loads copy number alteration data from BMEG."""

    oph = Ophion("http://bmeg.io")
    copy_list = {}

    for gene in gene_list:
        copy_list[gene] = {samp:0 for samp in sample_gids}

        for i in oph.query().has("gid", "gene:" + gene)\
                .incoming("segmentInGene").mark("segment")\
                .outgoing("segmentOfSample")\
                .has("gid", oph.within(list(sample_gids))).mark("sample")\
                .select(["sample", "segment"]).execute():
            copy_list[gene][i["sample"]["name"]] = i["segment"]["value"]

    return pd.DataFrame(copy_list)

