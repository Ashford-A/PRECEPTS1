
import os
base_dir = os.path.dirname(__file__)

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.features.cohorts import VariantCohort

import numpy as np
import synapseclient
import pickle
from itertools import combinations as combn


def main(argv):
    """Runs the experiment."""

    print(argv)
    out_path = os.path.join(base_dir, 'output', argv[0], argv[1])
    coh_lbl = 'TCGA-{}'.format(argv[0])

    train_cutoff = 12
    test_cutoff = 4

    syn = synapseclient.Synapse()
    syn.login()
    cdata = VariantCohort(syn, cohort=coh_lbl, mut_genes=[argv[1]],
                          mut_levels=['Gene', 'Form', 'Exon', 'Location'],
                          cv_seed=99)

    sub_mtypes = cdata.train_mut.treetypes(
        sub_levels=['Gene', 'Form', 'Exon'], min_size=train_cutoff)
    sub_mtypes |= cdata.train_mut.subtypes(
        sub_levels=['Location'], min_size=train_cutoff)

    use_mtypes = [mtype for mtype in sub_mtypes if
                  np.sum(cdata.test_pheno(mtype)) >= test_cutoff]
    print(len(use_mtypes))

    pickle.dump(use_mtypes, open(os.path.join(out_path, 'tmp/mtype_list.p'),
                                 'wb'))


if __name__ == "__main__":
    main(sys.argv[1:])

