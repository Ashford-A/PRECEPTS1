
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

    # finds the sub-types satisfying the sample frequency criterion, starting
    # with the ones that are combinations of individual branches
    use_mtypes = set()
    use_sampsets = set()
    comb_size = 0

    while len(use_mtypes) < 1000 and comb_size <= 5:
        comb_size += 1

        sub_mtypes = cdata.train_mut.combtypes(
            comb_sizes=(comb_size, ), min_type_size=freq_cutoff)
        if len(sub_mtypes) < 800:

            for mtype in sub_mtypes.copy():
                mtype_sampset = frozenset(mtype.get_samples(cdata.train_mut))

                if mtype_sampset in use_sampsets:
                    print("Removing functionally duplicate MuType {}"
                          .format(mtype))
                    sub_mtypes.remove(mtype)

                else:
                    use_sampsets.update(mtype_sampset)

            use_mtypes |= sub_mtypes
        
        else:
            break

        print("Found {} sub-types that are combinations of {} branch(es)."
              .format(len(sub_mtypes), comb_size))

    print("Using {} sub-types of branch combinations."
          .format(len(use_mtypes)))

    # adds the subtypes that are combinations of branches corresponding to
    # particular locations shared between different mutation forms
    comb_size = 0
    while len(use_mtypes) < 1500 and comb_size <= 10:
        comb_size += 1

        sub_mtypes = cdata.train_mut.combtypes(
            sub_levels=['Gene', 'Exon', 'Location'],
            comb_sizes=(comb_size, ), min_type_size=freq_cutoff)
        if len(sub_mtypes) < 400:

            for mtype in sub_mtypes.copy():
                mtype_sampset = frozenset(mtype.get_samples(cdata.train_mut))

                if mtype_sampset in use_sampsets:
                    print("Removing functionally duplicate MuType {}"
                          .format(mtype))
                    sub_mtypes.remove(mtype)

                else:
                    use_sampsets.update(mtype_sampset)

            use_mtypes |= sub_mtypes

        else:
            break

        print("Found {} sub-types that are combinations of {} exon/hotspot "
              "branch(es).".format(len(sub_mtypes), comb_size))

    print("Using {} total sub-types of branch and exon/hotspot combinations."
          .format(len(use_mtypes)))

    # save the list of sub-types to file
    pickle.dump(list(use_mtypes),
                open(os.path.join(out_path, 'tmp/mtype_list.p'), 'wb'))


if __name__ == "__main__":
    main(sys.argv[1:])

