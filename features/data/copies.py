
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


def get_copies_firehose(cohort, data_dir, discrete=True):
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
    copy_tars = glob.glob(os.path.join(
        data_dir, "analyses__2016_01_28", cohort, "20160128",
        '*CopyNumber_Gistic2.Level_4.*tar.gz'
        ))

    if len(copy_tars) > 1:
        raise IOError("Multiple GISTIC copy number tarballs found "
                      "for cohort {} in directory {} !".format(
                          cohort, data_dir))

    if len(copy_tars) == 0:
        raise IOError("No normalized GISTIC copy number tarballs found "
                      "for cohort {} in directory {} !".format(
                          cohort, data_dir))

    if discrete:
        fl_name = 'all_thresholded.by_genes.txt'
    else:
        fl_name = 'all_data_by_genes.txt'

    copy_tar = tarfile.open(copy_tars[0])
    copy_indx = [i for i, memb in enumerate(copy_tar.getmembers())
                 if fl_name in memb.get_info()['name']]

    # ensures only one file in the tarball contains CNA data
    if len(copy_indx) == 0:
        raise IOError("No thresholded CNA files found in the tarball!")
    elif len(copy_indx) > 1:
        raise IOError("Multiple thresholded CNA files found in the tarball!")

    copy_fl = copy_tar.extractfile(copy_tar.getmembers()[copy_indx[0]])
    copy_data = pd.read_csv(BytesIO(copy_fl.read()),
                            sep='\t', index_col=0, engine='python')
 
    copy_data = copy_data.iloc[:, 2:].transpose()
    copy_data.index = ["-".join(x[:4])
                       for x in copy_data.index.str.split('-')]

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

