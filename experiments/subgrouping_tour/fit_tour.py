
from ..subgrouping_tour import cis_lbls
from ..utilities.handle_input import safe_load
from ..utilities.pipeline_setup import get_task_count
from ..utilities.classifiers import *

import os
import argparse
import dill as pickle
import random


def main():
    parser = argparse.ArgumentParser(
        'fit_tour',
        description="Runs a portion of an experiment's classification tasks."
        )

    parser.add_argument('classif', type=str,
                        help="a classifier in HetMan.predict.classifiers")
    parser.add_argument('use_dir', type=str)

    parser.add_argument('--task_id', type=int, default=0,
                        help='the subset of subtypes to assign to this task')
    parser.add_argument('--cv_id', type=int, default=0,
                        help='the subset of subtypes to assign to this task')

    args = parser.parse_args()
    setup_dir = os.path.join(args.use_dir, 'setup')
    task_count = get_task_count(args.use_dir)

    with open(os.path.join(setup_dir, "muts-list.p"), 'rb') as muts_f:
        mtype_list = pickle.load(muts_f)

    coh_path = os.path.join(setup_dir, "cohort-data.p.gz")
    cdata = safe_load(coh_path, retry_pause=41)
    clf = eval(args.classif)
    mut_clf = clf()

    # figure out which cohort samples will be used for tuning and testing the
    # classifier and which samples will be used for testing
    use_seed = 9073 + 97 * args.cv_id
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

    random.seed(10301)
    random.shuffle(mtype_list)

    # for each subtype, check if it has been assigned to this task
    for i, mtype in enumerate(mtype_list):
        if (i % task_count) == args.task_id:
            print("Testing {} ...".format(mtype))

            for cis_lbl in cis_lbls:
                ex_genes = cdata.get_cis_genes(cis_lbl, mut=mtype)

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
                out_pred[mtype][cis_lbl] = np.round(mut_clf.parse_preds(
                    mut_clf.predict_test(cdata, lbl_type='raw',
                                         exclude_feats=ex_genes)
                    ), 7)

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

