
import os
import sys
base_dir = os.path.dirname(__file__)
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.experiments.subvariant_transfer import *
from HetMan.experiments.subvariant_infer.setup_infer import Mcomb, ExMcomb
from HetMan.experiments.utilities.classifiers import *

import argparse
import dill as pickle


def main():
    """Runs the experiment."""

    parser = argparse.ArgumentParser(
        "Isolate the expression signature of mutation subtypes from their "
        "parent gene(s)' signature or that of a list of genes in a given "
        "TCGA cohort."
        )

    parser.add_argument('classif', type=str,
                        help="a classifier in HetMan.predict.classifiers")
    parser.add_argument('--use_dir', type=str, default=base_dir)

    parser.add_argument(
        '--task_count', type=int, default=10,
        help='how many parallel tasks the list of types to test is split into'
        )
    parser.add_argument('--task_id', type=int, default=0,
                        help='the subset of subtypes to assign to this task')

    args = parser.parse_args()
    setup_dir = os.path.join(args.use_dir, 'setup')

    with open(os.path.join(setup_dir, "cohort-data.p"), 'rb') as cdata_f:
        cdata = pickle.load(cdata_f)
    with open(os.path.join(setup_dir, "muts-list.p"), 'rb') as muts_f:
        mtype_list = pickle.load(muts_f)

    clf = eval(args.classif)
    clf.predict_proba = clf.calc_pred_labels
    mut_clf = clf()

    out_tune = {test: {smps: {par: None for par, _ in mut_clf.tune_priors}
                       for smps in ['All', 'Iso']}
                for test in mtype_list}
    out_inf = {test: {'All': None, 'Iso': None} for test in mtype_list}

    for i, (cohort, mtype) in enumerate(mtype_list):
        if (i % args.task_count) == args.task_id:
            print("Isolating {} in cohort {} ...".format(mtype, cohort))

            use_gene = mtype.subtype_list()[0][0]
            coh_samps = cdata.cohort_samps[cohort.split('_')[0]]
            base_samps = cdata.train_mut[use_gene].get_samples() & coh_samps
            use_chr = cdata.gene_annot[use_gene]['Chr']

            ex_genes = {gene for gene, annot in cdata.gene_annot.items()
                        if annot['Chr'] == use_chr}
            ex_samps = base_samps - mtype.get_samples(cdata.train_mut)

            mut_clf, cv_output = mut_clf.tune_coh(
                cdata, mtype, include_samps=coh_samps, exclude_genes=ex_genes,
                tune_splits=4, test_count=48, parallel_jobs=12
                )

            clf_params = mut_clf.get_params()
            for par, _ in mut_clf.tune_priors:
                out_tune[(cohort, mtype)]['All'][par] = clf_params[par]

            out_inf[(cohort, mtype)]['All'] = mut_clf.infer_coh(
                cdata, mtype, exclude_genes=ex_genes,
                force_test_samps=cdata.samples - coh_samps,
                infer_splits=120, infer_folds=4, parallel_jobs=12
                )

            mut_clf, cv_output = mut_clf.tune_coh(
                cdata, mtype, exclude_genes=ex_genes,
                exclude_samps=ex_samps | (cdata.samples - coh_samps),
                tune_splits=4, test_count=48, parallel_jobs=12
                )

            clf_params = mut_clf.get_params()
            for par, _ in mut_clf.tune_priors:
                out_tune[(cohort, mtype)]['Iso'][par] = clf_params[par]

            out_inf[(cohort, mtype)]['Iso'] = mut_clf.infer_coh(
                cdata, mtype, exclude_genes=ex_genes,
                force_test_samps=ex_samps | (cdata.samples - coh_samps),
                infer_splits=120, infer_folds=4, parallel_jobs=12
                )

        else:
            del(out_inf[(cohort, mtype)])
            del(out_tune[(cohort, mtype)])

    pickle.dump({'Infer': out_inf, 'Tune': out_tune, 'Clf': mut_clf},
                open(os.path.join(args.use_dir, 'output',
                                  "out_task-{}.p".format(args.task_id)),
                     'wb'))


if __name__ == "__main__":
    main()

