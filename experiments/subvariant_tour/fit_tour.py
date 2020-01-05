
import os
import sys
base_dir = os.path.dirname(__file__)
sys.path.extend([os.path.join(base_dir, '..', '..', '..')])

from HetMan.experiments.subvariant_tour import cis_lbls
from HetMan.experiments.subvariant_test.classifiers import *

import argparse
import dill as pickle
import time
import random


def get_excluded_genes(cis_lbl, cur_gene, gene_annot):
    if cis_lbl == 'None':
        ex_genes = set()
    elif cis_lbl == 'Self':
        ex_genes = {cur_gene}

    elif cis_lbl == 'Chrm':
        use_chr = gene_annot[cur_gene]['Chr']
        ex_genes = {gene for gene, annot in gene_annot.items()
                    if annot['Chr'] == use_chr}

    else:
        raise ValueError(
            "Unrecognized cis-masking label `{}`!".format(cis_lbl))

    return ex_genes


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
    parser.add_argument('--use_dir', type=str, default=base_dir)

    parser.add_argument(
        '--task_count', type=int, default=10,
        help='how many parallel tasks the list of types to test is split into'
        )

    parser.add_argument('--task_id', type=int, default=0,
                        help='the subset of subtypes to assign to this task')
    parser.add_argument('--cv_id', type=int, default=0,
                        help='the subset of subtypes to assign to this task')

    args = parser.parse_args()
    setup_dir = os.path.join(args.use_dir, 'setup')
    coh_path = os.path.join(args.use_dir, '..', '..', "cohort-data.p")
    with open(os.path.join(setup_dir, "muts-list.p"), 'rb') as muts_f:
        mtype_list = pickle.load(muts_f)

    clf = eval(args.classif)
    mut_clf = clf()
    use_seed = 9073 + 97 * args.cv_id
    cdata = None

    while cdata is None:
        try:
            with open(coh_path, 'rb') as cdata_f:
                cdata = pickle.load(cdata_f)

        except:
            print("Failed to load cohort data, trying again...")
            time.sleep(61)

    cdata_samps = sorted(cdata.get_samples())
    random.seed((args.cv_id // 4) * 7712 + 13)
    random.shuffle(cdata_samps)
    cdata.update_split(use_seed, test_samps=cdata_samps[(args.cv_id % 4)::4])

    out_pars = {mtype: {cis_lbl: {par: None for par, _ in mut_clf.tune_priors}
                        for cis_lbl in cis_lbls}
                for mtype in mtype_list}

    out_time = {mtype: {cis_lbl: dict() for cis_lbl in cis_lbls}
                for mtype in mtype_list}
    out_acc = {mtype: {cis_lbl: dict() for cis_lbl in cis_lbls}
               for mtype in mtype_list}

    out_pred = {mtype: {cis_lbl: None for cis_lbl in cis_lbls}
                for mtype in mtype_list}

    # for each subtype, check if it has been assigned to this task
    for i, mtype in enumerate(mtype_list):
        if (i % args.task_count) == args.task_id:
            print("Testing {} ...".format(mtype))

            for cis_lbl in cis_lbls:
                ex_genes = get_excluded_genes(cis_lbl, mtype.get_labels()[0],
                                              cdata.gene_annot)

                # tune the hyper-parameters of the classifier
                mut_clf, cv_output = mut_clf.tune_coh(
                    cdata, mtype, exclude_feats=ex_genes,
                    tune_splits=4, test_count=mut_clf.test_count,
                    parallel_jobs=8
                    )

                # save the tuned values of the hyper-parameters
                clf_params = mut_clf.get_params()
                for par, _ in mut_clf.tune_priors:
                    out_pars[mtype][cis_lbl][par] = clf_params[par]

                out_time[mtype][cis_lbl]['avg'] = cv_output['mean_fit_time']
                out_time[mtype][cis_lbl]['std'] = cv_output['std_fit_time']
                out_acc[mtype][cis_lbl]['avg'] = cv_output['mean_test_score']
                out_acc[mtype][cis_lbl]['std'] = cv_output['std_test_score']
                out_acc[mtype][cis_lbl]['par'] = cv_output['params']
 
                mut_clf.fit_coh(cdata, mtype, exclude_feats=ex_genes)
                out_pred[mtype][cis_lbl] = mut_clf.parse_preds(
                    mut_clf.predict_test(cdata, lbl_type='raw',
                                         exclude_feats=ex_genes)
                    )

        else:
            del(out_pars[mtype])
            del(out_time[mtype])
            del(out_acc[mtype])
            del(out_pred[mtype])

    with open(os.path.join(args.use_dir, 'output',
                           "out__cv-{}_task-{}.p".format(
                               args.cv_id, args.task_id)),
              'wb') as fl:
        pickle.dump({'Pred': out_pred, 'Pars': out_pars, 'Time': out_time,
                     'Acc': out_acc, 'Clf': mut_clf.__class__},
                    fl, protocol=-1)


if __name__ == "__main__":
    main()

