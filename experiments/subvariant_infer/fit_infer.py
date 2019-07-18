
import os
import sys
base_dir = os.path.dirname(__file__)
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.experiments.subvariant_infer import *
from HetMan.experiments.subvariant_infer.setup_infer import Mcomb, ExMcomb
from dryadic.learning.classifiers import *
from dryadic.features.mutations import MuType

import argparse
import dill as pickle


def main():
    """Runs the experiment."""

    parser = argparse.ArgumentParser(
        "Isolate the expression signature of mutation subtypes from their "
        "parent gene(s)' signature or that of a list of genes in a given "
        "TCGA cohort."
        )

    # positional arguments for which cohort of samples and which mutation
    # classifier to use for testing
    parser.add_argument('classif', type=str,
                        help="a classifier in HetMan.predict.classifiers")
    parser.add_argument('gene', type=str,
                        help="which gene's mutations to isolate against")
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

    out_tune = {mtype: {smps: {par: None for par, _ in mut_clf.tune_priors}
                        for smps in ['All', 'Iso']}
                for mtype in mtype_list}
    out_inf = {mtype: {'All': None, 'Iso': None} for mtype in mtype_list}

    # find the genes on the same chromosome as the gene whose mutations are
    # being isolated which will be removed from the features used for training
    base_samps = cdata.mtree.get_samples()
    use_chr = cdata.gene_annot[args.gene]['Chr']
    ex_genes = {gene for gene, annot in cdata.gene_annot.items()
                if annot['Chr'] == use_chr}

    # for each subtype, check if it has been assigned to this task
    for i, mtype in enumerate(mtype_list):
        if (i % args.task_count) == args.task_id:
            print("Isolating {} ...".format(mtype))
            ex_samps = base_samps - mtype.get_samples(cdata.mtree)

            # tune the hyper-parameters of the classifier
            mut_clf, cv_output = mut_clf.tune_coh(
                cdata, mtype, exclude_feats=ex_genes,
                tune_splits=8, test_count=48, parallel_jobs=8
                )

            # save the tuned values of the hyper-parameters
            clf_params = mut_clf.get_params()
            for par, _ in mut_clf.tune_priors:
                out_tune[mtype]['All'][par] = clf_params[par]

            out_inf[mtype]['All'] = mut_clf.infer_coh(
                cdata, mtype, exclude_feats=ex_genes,
                infer_splits=200, infer_folds=4, parallel_jobs=8
                )

            # tune the hyper-parameters of the classifier
            mut_clf, cv_output = mut_clf.tune_coh(
                cdata, mtype, exclude_feats=ex_genes, exclude_samps=ex_samps,
                tune_splits=8, test_count=48, parallel_jobs=8
                )

            # save the tuned values of the hyper-parameters
            clf_params = mut_clf.get_params()
            for par, _ in mut_clf.tune_priors:
                out_tune[mtype]['Iso'][par] = clf_params[par]

            out_inf[mtype]['Iso'] = mut_clf.infer_coh(
                cdata, mtype,
                exclude_feats=ex_genes, force_test_samps=ex_samps,
                infer_splits=200, infer_folds=4, parallel_jobs=8
                )

        else:
            del(out_inf[mtype])
            del(out_tune[mtype])

    pickle.dump(
        {'Infer': out_inf, 'Tune': out_tune, 'Clf': mut_clf},
        open(os.path.join(args.use_dir, 'output',
                          "out_task-{}.p".format(args.task_id)),
             'wb')
        )


if __name__ == "__main__":
    main()

