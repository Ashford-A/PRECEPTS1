
import os
base_dir = os.path.dirname(__file__)

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.features.expression import get_expr_firehose
from HetMan.features.variants import get_variants_mc3, MuType
from HetMan.features.cohorts import VariantCohort

import numpy as np
import synapseclient
import dill as pickle

from itertools import combinations as combn
from itertools import chain

from HetMan.experiments.utilities.test_cohort_mutypes import load_output

firehose_dir = '/home/exacloud/lustre1/CompBio/mgrzad/input-data/firehose'


def main(argv):
    """Runs the experiment."""

    # get the directory where results will be saved and the name of the
    # TCGA cohort we will be using
    print(argv)
    out_path = os.path.join(base_dir, 'output', argv[0], argv[1], 'portray')
    auc_cutoff = float(argv[2])

    out_data = load_output(os.path.join(
        base_dir, 'output', argv[0], argv[1], 'search'))
    portray_mtypes = out_data.loc[out_data.min(axis=1) > auc_cutoff, :].index

    print("Investigating {} sub-types that were found to have an AUC of at"
          "least {} during the search phase."
            .format(len(portray_mtypes), argv[2]))

    # save the list of sub-types to file
    pickle.dump(list(portray_mtypes),
                open(os.path.join(out_path, 'tmp/mtype_list.p'), 'wb'))


if __name__ == "__main__":
    main(sys.argv[1:])

