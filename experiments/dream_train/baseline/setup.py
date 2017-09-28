
"""Setting up the DREAM challenge baseline testing experiment.

Args:

Examples:

"""

import os
base_dir = os.path.dirname(__file__)

import sys
sys.path.extend([os.path.join(base_dir, '../../../..')])

from HetMan.features.cohorts import DreamCohort

import synapseclient
import dill as pickle


def main(argv):
    """Runs the experiment."""

    # get the directory where results will be saved and the name of the
    # TCGA cohort we will be using
    print(argv)
    out_path = os.path.join(base_dir, 'output', argv[0], argv[1], argv[2])

    # load the expression data and the gene's mutation data
    # for the given cohort
    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = ("/home/exacloud/lustre1/CompBio/"
                                "mgrzad/input-data/synapse")

    syn.login()
    cdata = DreamCohort(syn, argv[0], cv_prop=1.0)

    # save the list of sub-types to file
    pickle.dump(list(sorted(cdata.train_prot.columns)),
                open(os.path.join(out_path, 'tmp/gene_list.p'), 'wb'))


if __name__ == "__main__":
    main(sys.argv[1:])

