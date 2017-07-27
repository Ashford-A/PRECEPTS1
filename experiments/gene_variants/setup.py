
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

from HetMan.features.cohorts import VariantCohort, MutCohort

import numpy as np
import synapseclient
import pickle

from itertools import combinations as combn
from itertools import chain
from functools import reduce

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
    cdata = VariantCohort(
        syn, cohort=coh_lbl, mut_genes=[argv[1]],
        mut_levels=['Gene', 'Form_base', 'Exon', 'Location'],
        cv_prop=1.0
        )

    # finds the sub-types satisfying the sample frequency criterion
    sub_mtypes = cdata.train_mut.subtypes(min_size=freq_cutoff)
    sub_mtypes |= set(
        reduce(lambda x, y: x | y, mtypes)
        for mtypes in chain(combn(sub_mtypes, 2), combn(sub_mtypes, 3))
        )
    sub_mtypes |= cdata.train_mut.treetypes(
        min_size=20, sub_levels=['Gene', 'Form_base'])

    exon_mtypes = cdata.train_mut.subtypes(
        min_size=10, sub_levels=['Gene', 'Exon'])
    sub_mtypes |= exon_mtypes
    sub_mtypes |= set(
        reduce(lambda x, y: x | y, mtypes)
        for mtypes in chain(combn(exon_mtypes, 2), combn(exon_mtypes, 3))
        )

    loc_mtypes = cdata.train_mut.subtypes(
        min_size=10, sub_levels=['Gene', 'Location'])
    sub_mtypes |= loc_mtypes
    sub_mtypes |= set(reduce(lambda x, y: x | y, mtypes)
                      for mtypes in combn(loc_mtypes, 2))

    # save the list of sub-types to file
    print(len(sub_mtypes))
    pickle.dump(list(sub_mtypes),
                open(os.path.join(out_path, 'tmp/mtype_list.p'), 'wb'))


if __name__ == "__main__":
    main(sys.argv[1:])

