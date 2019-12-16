
"""Loading and processing copy number alteration datasets.

This module contains functions for retrieving CNA data and processing it into
formats suitable for use in machine learning pipelines.

Author: Michal Grzadkowski <grzadkow@ohsu.edu>

"""

import pandas as pd

import os
import glob

import tarfile
from io import BytesIO


def get_copies_firehose(cohort, data_dir, discrete=True, normalize=False):
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
        "*CopyNumber_Gistic2.Level_4.*tar.gz"
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
        fl_name = "all_thresholded.by_genes.txt"
    else:
        fl_name = "all_data_by_genes.txt"

    copy_tar = tarfile.open(copy_tars[0])
    gene_indx = [i for i, memb in enumerate(copy_tar.getmembers())
                 if fl_name in memb.get_info()['name']]

    # ensures only one file in the tarball contains CNA data
    if len(gene_indx) == 0:
        raise IOError("No thresholded CNA files found in the tarball!")
    elif len(gene_indx) > 1:
        raise IOError("Multiple thresholded CNA files found in the tarball!")

    gene_fl = copy_tar.extractfile(copy_tar.getmembers()[gene_indx[0]])
    gene_data = pd.read_csv(BytesIO(gene_fl.read()),
                            sep='\t', index_col=0, engine='python')

    gene_data = gene_data.iloc[:, 2:].transpose()
    gene_data.index = ['-'.join(smp[:4])
                       for smp in gene_data.index.str.split('-')]
    gene_data.columns.name = None

    if normalize:
        ctf_indx = [i for i, memb in enumerate(copy_tar.getmembers())
                    if 'sample_cutoffs.txt' in memb.get_info()['name']]
        ctf_fl = copy_tar.extractfile(copy_tar.getmembers()[ctf_indx[0]])

        ctf_data = pd.read_csv(BytesIO(ctf_fl.read()), sep='\t',
                               index_col=0, comment='#', engine='python')
        ctf_data.index = ["-".join(smp[:4])
                          for smp in ctf_data.index.str.split('-')]

        for smp in set(gene_data.index) & set(ctf_data.index):
            smp_vals = gene_data.loc[smp] 
            smp_vals[smp_vals < 0] /= -ctf_data.loc[smp, 'Low']
            smp_vals[smp_vals > 0] /= ctf_data.loc[smp, 'High']
            gene_data.loc[smp] = smp_vals.round(3)

    regn_indx = [i for i, memb in enumerate(copy_tar.getmembers())
                 if "all_lesions.conf_" in memb.get_info()['name']]
    regn_fl = copy_tar.extractfile(copy_tar.getmembers()[regn_indx[0]])

    regn_data = pd.read_csv(BytesIO(regn_fl.read()),
                            sep='\t', index_col=0, engine='python')
    regn_data.Descriptor = regn_data.Descriptor.str.replace("\s+$", "")
    contn_indx = regn_data.index.str.match(".* - CN values$")
    regn_mat = regn_data.iloc[contn_indx ^ discrete, 8:-1]

    regn_mat.index = ["{}({})".format(regn_data.loc[lbl, 'Descriptor'],
                                      lbl[:3])
                      for lbl in regn_mat.index]

    if discrete:
        regn_mat = regn_mat.astype(int)
        del_indx = regn_mat.index.str.match(".*\(Del\)$")
        regn_mat.loc[del_indx] *= -1

    regn_mat.columns = ['-'.join(smp[:4])
                        for smp in regn_mat.columns.str.split('-')]
    regn_mat.index.name = None

    if discrete:
        carm_data = pd.DataFrame(columns=regn_mat.columns)

    else:
        carm_indx = [i for i, memb in enumerate(copy_tar.getmembers())
                     if "broad_values_by_arm.txt" in memb.get_info()['name']]
        carm_fl = copy_tar.extractfile(copy_tar.getmembers()[carm_indx[0]])
        carm_data = pd.read_csv(BytesIO(carm_fl.read()),
                                sep='\t', index_col=0, engine='python')

        carm_data = carm_data.loc[carm_data.index.str.match("[0-9]+[p|q]")]
        carm_data.columns = ['-'.join(smp[:4])
                             for smp in carm_data.columns.str.split('-')]

    carm_data.index.name = None
    copy_tar.close()

    return pd.concat([gene_data, regn_mat.T, carm_data.T],
                     axis=1, join='inner')


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

