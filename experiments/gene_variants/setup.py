
"""Setting up the gene sub-variants experiment.

This script finds all of the sub-variants for a gene in a TCGA cohort that
we want to find downstream expression effects for.

Args:
    setup.py <cohort> <gene>

Examples:
    setup.py BRCA TP53
    setup.py UCEC PTEN
    setup.py SKCM TTN

"""

import os
base_dir = os.path.dirname(__file__)

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.features.cohorts import VariantCohort

import numpy as np
from itertools import combinations as combn
import synapseclient
import pickle

# how many samples must contain a mutation for us to consider it?
freq_cutoff = 20


def main(argv):
    """Runs the experiment."""

    # get the directory where results will be saved and the name of the
    # TCGA cohort we will be using
    print(argv)
    out_path = os.path.join(base_dir, 'output', argv[0], argv[1])
    coh_lbl = 'TCGA-{}'.format(argv[0])

    # load the expression data and the gene's mutation data
    # for the given cohort
    syn = synapseclient.Synapse()
    syn.login()
    cdata = VariantCohort(syn, cohort=coh_lbl, mut_genes=[argv[1]],
                          mut_levels=['Gene', 'Form', 'Exon', 'Location'],
                          cv_prop=1.0)

    # finds the sub-types satisfying the sample frequency criterion
    sub_mtypes = cdata.train_mut.treetypes(
        sub_levels=['Gene', 'Form', 'Exon'], min_size=freq_cutoff)
    sub_mtypes |= cdata.train_mut.subtypes(
        sub_levels=['Location'], min_size=freq_cutoff)

    # save the list of sub-types to file
    print(len(sub_mtypes))
    pickle.dump(list(sub_mtypes),
                open(os.path.join(out_path, 'tmp/mtype_list.p'), 'wb'))


if __name__ == "__main__":
    main(sys.argv[1:])

