
"""Loading and processing proteomic datasets.

Author: Michal Grzadkowski <grzadkow@ohsu.edu>

"""

import pandas as pd


def get_prot_cptac(syn, cohort, source=None):
    """Get the CPTAC proteomic data used in the DREAM challenge.

    Args:
        syn (synapseclient.Synapse): A logged-into Synapse instance.
        cohort (str): A TCGA cohort included in the challenge.

    Examples:
        >>> import synapseclient
        >>> syn = synapseclient.Synapse()
        >>>
        >>> syn.cache.cache_root_dir = (
        >>>     '/home/exacloud/lustre1/'
        >>>     'share_your_data_here/precepts/synapse'
        >>>     )
        >>> syn.login()
        >>>
        >>> get_prot_cptac(syn, "BRCA")
        >>> get_prot_cptac(syn, "OV", source="PNNL")

    """

    syn_ids = {'BRCA': '11328678',
               'OV': {'JHU': '10514980', 'PNNL': '10514979'}}

    if isinstance(syn_ids[cohort], dict):
        use_id = syn_ids[cohort][source]
    else:
        use_id = syn_ids[cohort]

    prot_data = pd.read_csv(syn.get("syn{}".format(use_id)).path,
                            sep='\t', index_col=0).transpose()

    return prot_data

