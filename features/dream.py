
"""Loading data for the NCI-CPTAC DREAM Proteogenomics Challenge.

Author: Michal Grzadkowski <grzadkow@ohsu.edu>

"""

import pandas as pd
import synapseutils


# Synapse IDs for each combination of cohort and -omic dataset type
syn_ids = {
    'BRCA': {'rna': '10139529',
             'cna': '10139527',
             'prot': '10139538'},
    
    'OV': {'rna': '10535396',
           'cna': '10139531',
           'prot': {'JHU': '10514980', 'PNNL': '10514979'}}
    }


def get_dream_data(syn, cohort, omic_type, source=None):
    """Retrieves a particular -omic dataset used in the challenge.

    Args:
        syn (synapseclient.Synapse): A logged-into Synapse instance.
        cohort (str): A TCGA cohort included in the challenge.
        omic_type (str): A type of -omics used in the challenge.
            Note that multiple -omic types can be downloaded by listing
            the -omic types in a single string, separated by a '+'.

    Examples:
        >>> import synapseclient
        >>> syn = synapseclient.Synapse()
        >>> syn.login()
        >>>
        >>> get_dream_data(syn, "BRCA", "rna")
        >>> get_dream_data(syn, "OV", "cna")
        >>> get_dream_data(syn, "BRCA", "rna+cna")

    """
    syn_manifest = synapseutils.syncFromSynapse(
        syn, "syn10139523", ifcollision='overwrite.local')

    # if we want to use multiple -omic datasets, get Synapse ids
    # for all of them...
    if '+' in omic_type:
        omic_type = omic_type.split('+')
        dream_ids = [syn_ids[cohort][omic_tp]
                     for omic_tp in omic_type]

    # ...otherwise, get the Synapse id for the one dataset
    else:
        dream_ids = syn_ids[cohort][omic_type]
        omic_type = [omic_type]
        dream_ids = [dream_ids]

    dream_ids = [dream_id if isinstance(dream_id, str) else dream_id[source]
                 for dream_id in dream_ids]

    # read in the -omic dataset(s) and merge them according to sample ID
    dream_data = pd.concat(
        {omic_tp: pd.read_csv(syn.get("syn{}".format(dream_id)).path,
                              sep='\t', index_col=0).transpose()
         for dream_id, omic_tp in zip(dream_ids, omic_type)},
        join='inner', axis=1
        )

    # convert the pandas MultiIndex of the merged datasets into a flat
    # list of strings
    dream_data.columns = ["__".join(col)
                          for col in dream_data.columns.values]

    return dream_data

