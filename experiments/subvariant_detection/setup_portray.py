
import os
base_dir = os.path.dirname(__file__)

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])

import HetMan.experiments.utilities as utils
import dill as pickle
import argparse


def main():
    """Runs the experiment."""

    parser = argparse.ArgumentParser(
        description="Set up portrayal of sub-type signatures."
        )

    # positional command line arguments
    parser.add_argument('cohort', type=str, help='a TCGA cohort')
    parser.add_argument('classif', type=str,
                        help='a classifier ijn HetMan.predict.classifiers')

    parser.add_argument(
        '--auc_cutoff', type=float, default=0.8,
        help='AUC classification performance threshold'
        )

    parser.add_argument('--verbose', '-v', action='store_true',
                        help='turns on diagnostic messages')

    # get the directory where results will be saved and the name of the
    # TCGA cohort we will be using
    args = parser.parse_args()
    out_path = os.path.join(base_dir, 'output',
                            args.cohort, args.classif, 'portray')

    # loads information about the sub-types found in the search phase, finds
    # the sub-types that pass the AUC criterion
    out_data = utils.test_output(os.path.join(
        base_dir, 'output', args.cohort, args.classif, 'search'))
    portray_mtypes = out_data.loc[
        out_data.min(axis=1) > args.auc_cutoff, :].index

    print("Investigating {} sub-types that were found to have an AUC of at "
          "least {} during the search phase."
            .format(len(portray_mtypes), args.auc_cutoff))

    # save the list of sub-types whose expression profile is to be portrayed
    # to file for use by further scripts
    pickle.dump(list(portray_mtypes),
                open(os.path.join(out_path, 'tmp/mtype_list.p'), 'wb'))


if __name__ == "__main__":
    main()

