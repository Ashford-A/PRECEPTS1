
"""Loading and processing copy number alteration datasets.

This module contains functions for retrieving CNA data and processing it into
formats suitable for use in machine learning pipelines.

Author: Michal Grzadkowski <grzadkow@ohsu.edu>

"""

import pandas as pd
from ophion import Ophion

import os
import glob

import tarfile
from io import BytesIO


def get_copies_firehose(cohort, data_dir):
    """Loads gene-level copy number alteration data downloaded from Firehose.

    Args:
        cohort (str): A TCGA cohort available in Broad Firehose.
        data_dir (str): A local directory where the data has been downloaded.

    Returns:
        copy_data (pandas DataFrame), shape = [n_samps, n_genes]

    Examples:
        >>> copy_data = get_copies_firehose(
        >>>     'BRCA', '/home/users/grzadkow/compbio/input-data/firehose')
        >>> copy_data = get_copies_firehose('STAD', '../input-data')

    """
    copy_tar = tarfile.open(glob.glob(os.path.join(
        data_dir, "analyses__2016_01_28", cohort, "20160128",
        '*CopyNumber_Gistic2.Level_4.*tar.gz'
        ))[0])

    copy_fl = copy_tar.extractfile(copy_tar.getmembers()[-5])
    copy_data = pd.read_csv(BytesIO(copy_fl.read()),
                            sep='\t', index_col=0, engine='python')
    copy_data = copy_data.iloc[:, 2:].transpose().fillna(0.0)

    return copy_data


def get_copies_bmeg(cohort, gene_list):
    """Loads gene-level copy number data from BMEG.

    Args:
        cohort (str): The name of an individualCohort vertex in BMEG.

    Returns:
        copy_data (pandas DataFrame of float), shape = [n_samps, n_feats]

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

