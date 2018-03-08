
import os
base_dir = os.path.dirname(__file__)

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.features.cohorts.tcga import MutationCohort as TCGAcohort
from HetMan.features.cohorts.icgc import MutationCohort as ICGCcohort
from HetMan.features.variants import MuType
from HetMan.predict.basic.classifiers import *

import argparse
import synapseclient
from math import ceil
import dill as pickle

icgc_data_dir = '/home/exacloud/lustre1/share_your_data_here/precepts/ICGC'
toil_dir = "/home/exacloud/lustre1/CompBio/mgrzad/input-data/toil/processed"


def main():
    """Runs the experiment."""

    parser = argparse.ArgumentParser(
        description=("Test a classifier's ability to create a mutation "
                     "signature for a gene that can be transferred from a "
                     "TCGA cohort to ICGC PACA-AU.")
        )

    parser.add_argument('classif', type=str,
                        help='a classifier in HetMan.predict.classifiers')
    parser.add_argument('cv_id', type=int,
                        help='a random seed used for cross-validation')
    parser.add_argument('task_id', type=int,
                        help=('the subset of TCGA cohorts and mutated genes '
                              'to assign to this task'))

    parser.add_argument(
        '--tune_splits', type=int, default=4,
        help='how many training cohort splits to use for tuning'
        )
    parser.add_argument(
        '--test_count', type=int, default=24,
        help='how many hyper-parameter values to test in each tuning split'
        )
    parser.add_argument(
        '--parallel_jobs', type=int, default=8,
        help='how many parallel CPUs to allocate the tuning tests across'
        )

    parser.add_argument('--verbose', '-v', action='store_true',
                        help='turns on diagnostic messages')

    args = parser.parse_args()
    if args.verbose:
        print("Starting ICGC transfer test with classifier {} for "
              "cross-validation ID {} and task ID {} ...".format(
                  args.classif, args.cv_id, args.task_id))

    cohort_genes = sorted(pickle.load(
        open(os.path.join(base_dir, 'setup', 'cohort_genes.p'), 'rb')))

    test_count = ceil(len(cohort_genes) / 5)
    cohort_genes = [x for i, x in enumerate(cohort_genes)
                    if i // test_count == args.task_id]

    use_cohorts = set(coh for coh, _ in cohort_genes)
    mut_clf = eval(args.classif)
    out_acc = {cohort: dict() for cohort in use_cohorts}

    cdata_icgc = ICGCcohort('PACA-AU', icgc_data_dir, mut_genes=None,
                            samp_cutoff=[1/12, 11/12], cv_prop=1.0)

    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = ("/home/exacloud/lustre1/CompBio/mgrzad"
                                "/input-data/synapse")
    syn.login()
    
    for cohort in use_cohorts:
        cur_genes = [gn for coh, gn in cohort_genes if coh == cohort]

        tcga_cdata = TCGAcohort(
            cohort=cohort, mut_genes=cur_genes, mut_levels=['Gene'],
            expr_source='toil', expr_dir=toil_dir, var_source='mc3', syn=syn,
            collapse_txs=True, cv_prop=0.75, cv_seed=(args.cv_id - 37) * 101
            )

        if args.verbose:
            print("Loaded mutations for {} genes in cohort {} with "
                  "{} samples.".format(len(cur_genes), cohort,
                                       len(tcga_cdata.samples)))

        for gene in cur_genes:
            if args.verbose:
                print("Testing {} ...".format(gene))

            clf = mut_clf()
            use_genes = (cdata_icgc.genes & tcga_cdata.genes) - {gene}
            use_mtype = MuType({('Gene', gene): None})

            clf.tune_coh(tcga_cdata, use_mtype, include_genes=use_genes,
                         tune_splits=args.tune_splits,
                         test_count=args.test_count,
                         parallel_jobs=args.parallel_jobs)

            clf.fit_coh(tcga_cdata, use_mtype, include_genes=use_genes)
            out_acc[cohort][gene] = clf.eval_coh(
                cdata_icgc, use_mtype, include_genes=use_genes,
                use_train=True
                )

    out_file = os.path.join(base_dir, 'output', args.classif,
                            'out__cv-{}_task-{}.p'.format(
                                args.cv_id, args.task_id)
                            )

    pickle.dump({'Acc': out_acc,
                 'Info': {'TuneSplits': args.tune_splits,
                          'TestCount': args.test_count,
                          'ParallelJobs': args.parallel_jobs}},
                open(out_file, 'wb'))


if __name__ == "__main__":
    main()
