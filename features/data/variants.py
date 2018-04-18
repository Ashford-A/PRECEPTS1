
"""Loading and processing variant datasets.

This file contains functions and classes for loading, processing, and storing
mutations such as SNPs, indels, and frameshifts in formats suitable for use
in machine learning pipelines.

See Also:
    :module:`.utils`: Utilities common across many types of features.
    :module:`.copies`: Dealing with copy number alterations.

Author: Michal Grzadkowski <grzadkow@ohsu.edu>

"""

import numpy as np
import pandas as pd

import tarfile
import os
import glob

from io import BytesIO
import json

from re import sub as gsub
from functools import reduce


# .. functions for loading mutation data from external data sources ..
def get_variants_mc3(syn):
    """Reads ICGC mutation data from the MC3 synapse file.

    Args:
        syn (Synapse): A logged-in synapseclient instance.

    Returns:
        muts (pandas DataFrame), shape = [n_mutations, mut_levels + 1]
            An array of mutation data, with a row for each mutation
            appearing in an individual sample.

    Examples:
        >>> import synapseclient
        >>> syn = synapseclient.Synapse()
        >>> syn.login()
        >>> muts = get_variants_mc3(syn)

    """
    mc3 = syn.get('syn7824274')

    # defines which mutation annotation MAF columns to use
    use_cols = [0, 8, 15, 36, 37, 38, 72]
    use_names = ['Gene', 'Form', 'Sample',
                 'Protein', 'Transcript', 'Exon', 'PolyPhen']

    # imports mutation data into a DataFrame, parses TCGA sample barcodes
    # and PolyPhen scores
    muts = pd.read_csv(mc3.path, usecols=use_cols, sep='\t', header=None,
                       names=use_names, comment='#', skiprows=1)
    muts['Sample'] = [reduce(lambda x, y: x + '-' + y, s.split('-', 4)[:4])
                      for s in muts['Sample']]
    muts['PolyPhen'] = [gsub('\)$', '', gsub('^.*\(', '', x))
                        if x != '.' else 0 for x in muts['PolyPhen']]

    return muts


def get_variants_firehose(cohort, data_dir):
    """Gets variant calls that have been downloaded from Firehose.

    Args:
        cohort (str): A TCGA cohort available in Broad Firehose.
        data_dir (str): A local directory where the data has been downloaded.

    Returns:

    Examples:

    """
    mut_tar = tarfile.open(glob.glob(os.path.join(
        data_dir, "stddata__2016_01_28", cohort, "20160128",
        "*Mutation_Packager_Oncotated_Calls.Level_3*tar.gz"
        ))[0])

    mut_list = []
    for mut_fl in mut_tar.getmembers():

        try:
            mut_tbl = pd.read_csv(
                BytesIO(mut_tar.extractfile(mut_fl).read()),
                sep='\t', skiprows=4, usecols=[0, 8, 15, 37, 41],
                names=['Gene', 'Form', 'Sample', 'Exon', 'Protein']
                )
            mut_list += [mut_tbl]

        except:
            print("Skipping mutations for {}".format(mut_fl))
        
    mut_data = pd.concat(mut_list)
    mut_data['Sample'] = ["-".join(x[:4])
                          for x in mut_data['Sample'].str.split('-')]

    return mut_data


def get_variants_icgc(cohort, data_dir):
    """Gets variants for an ICGC cohort that have been downloaded locally.

    Args:
        cohort (str): The name of an ICGC cohort downloaded locally.
        data_dir (str): The path where the ICGC data has been downloaded.

    Returns:
        mut_data (:obj:`pd.DataFrame`, shape = [n_mutations, 5])

    """

    mut_data = pd.read_csv(
        os.path.join(data_dir, cohort, 'simple_somatic_mutation.open.tsv.gz'),
        sep='\t', skiprows=1, usecols=[5, 13, 25, 28, 29],
        names=['Sample', 'Type', 'Form', 'Gene', 'Transcript']
        )

    return mut_data


def get_variants_bmeg(sample_list, gene_list, mut_fields=("term", )):
    """Gets variants from BMEG."""

    oph = Ophion("http://bmeg.io")
    mut_list = {samp: {} for samp in sample_list}
    gene_lbls = ["gene:" + gn for gn in gene_list]

    print(oph.query().has("gid", "biosample:" + sample_list[0])
          .incoming("variantInBiosample")
          .outEdge("variantInGene").mark("variant")
          .inVertex().has("gid", oph.within(gene_lbls)).count().execute())
          # .mark("gene").select(["gene", "variant"]).count().execute())

    for samp in sample_list:
        for i in oph.query().has("gid", "biosample:" + samp)\
                .incoming("variantInBiosample")\
                .outEdge("variantInGene").mark("variant")\
                .inVertex().has("gid", oph.within(gene_lbls))\
                .mark("gene").select(["gene", "variant"]).execute():
            dt = json.loads(i)
            gene_name = dt["gene"]["properties"]["symbol"]
            mut_list[samp][gene_name] = {
                k: v for k, v in dt["variant"]["properties"].items()
                if k in mut_fields}

    mut_table = pd.DataFrame(mut_list)

    return mut_table
