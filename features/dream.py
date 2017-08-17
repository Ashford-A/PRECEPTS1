
"""Loading data for the NCI-CPTAC DREAM Proteogenomics Challenge.

Author: Michal Grzadkowski <grzadkow@ohsu.edu>

"""

import pandas as pd


syn_ids = {
    'BRCA': {'rna': '10139529',
             'cna': '10139527',
             'prot': '10139538'},
    
    'OV': {'rna': '10139533',
           'cna': '10139531',
           'prot': ['10290694', '10290695']}
    }


def get_dream_data(syn, cohort, omic_type):
    """Retrieves a particular -omic dataset used in the challenge.

    Args:
        syn (synapseclient.Synapse): A logged-into Synapse instance.
        cohort (str): A TCGA cohort included in the challenge.
        omic_type (str): A type of -omics used in the challenge.

    Examples:
        >>> import synapseclient
        >>> syn = synapseclient.Synapse()
        >>> syn.login()
        >>>
        >>> get_dream_data(syn, "BRCA", "rna")
        >>> get_dream_data(syn, "OV", "cna")

    """
    dream_ids = syn_ids[cohort][omic_type]

    if isinstance(dream_ids, str):
        dream_data = pd.read_csv(syn.get("syn{}".format(dream_ids)).path,
                                 sep='\t', index_col=0).transpose()
    else:
        pass

    return dream_data

