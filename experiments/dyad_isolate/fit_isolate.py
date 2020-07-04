
from ..utilities.handle_input import safe_load
from ..utilities.mutations import pnt_mtype, shal_mtype, ExMcomb
from ..subvariant_isolate.classifiers import *

import os
import argparse
import dill as pickle
import random

from functools import reduce
from operator import or_


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
    parser.add_argument('use_dir', type=str)

    parser.add_argument(
        '--task_count', type=int, default=1,
        help='how many parallel tasks the list of types to test is split into'
        )

    parser.add_argument('--task_id', type=int, default=0,
                        help='the subset of subtypes to assign to this task')
    parser.add_argument('--cv_id', type=int, default=0,
                        help='the subset of subtypes to assign to this task')

    args = parser.parse_args()
    setup_dir = os.path.join(args.use_dir, 'setup')

    with open(os.path.join(setup_dir, "muts-list.p"), 'rb') as muts_f:
        mtype_list = pickle.load(muts_f)
    cdata = safe_load(os.path.join(setup_dir, "cohort-data.p.gz"),
                      retry_pause=31)

    use_mtree = tuple(cdata.mtrees.values())[0]
    clf = eval(args.classif)
    mut_clf = clf()

    use_seed = 13101 + 103 * args.cv_id
    cdata_samps = sorted(cdata.get_samples())
    random.seed((args.cv_id // 4) * 3901 + 23)
    random.shuffle(cdata_samps)
    cdata.update_split(use_seed, test_samps=cdata_samps[(args.cv_id % 4)::4])

    out_pars = {mtype: {smps: {par: None for par, _ in mut_clf.tune_priors}
                        for smps in ['All', 'Iso', 'IsoShal']}
                for mtype in mtype_list}

    out_time = {mtype: {smps: dict() for smps in ['All', 'Iso', 'IsoShal']}
                for mtype in mtype_list}
    out_acc = {mtype: {smps: dict() for smps in ['All', 'Iso', 'IsoShal']}
               for mtype in mtype_list}
    out_pred = {mtype: {smps: None for smps in ['All', 'Iso', 'IsoShal']}
                for mtype in mtype_list}

    # for each subtype, check if it has been assigned to this task
    for i, mtype in enumerate(mtype_list):
        if (i % args.task_count) == args.task_id:
            print("Isolating {} ...".format(mtype))

            cur_genes = mtype.get_labels()
            ex_genes = cdata.get_cis_genes('Chrm', cur_genes=cur_genes)
            mut_samps = reduce(or_, [use_mtree[gene].get_samples()
                                     for gene in cur_genes])

            shal_samps = reduce(
                or_,
                [ExMcomb(pnt_mtype, shal_mtype).get_samples(use_mtree[gene])
                 for gene in cur_genes]
                )

            mtype_samps = {
                samp for samp, phn in zip(cdata.get_train_samples(),
                                          cdata.train_pheno(mtype))
                if phn
                }

            ex_dict = {'All': set(), 'Iso': mut_samps - mtype_samps,
                       'IsoShal': mut_samps - mtype_samps - shal_samps}

            for ex_lbl, ex_samps in ex_dict.items():
                mut_clf, cv_output = mut_clf.tune_coh(
                    cdata, mtype, exclude_feats=ex_genes,
                    exclude_samps=ex_samps, tune_splits=4,
                    test_count=mut_clf.test_count, parallel_jobs=8
                    )

                # save the tuned values of the hyper-parameters
                clf_params = mut_clf.get_params()
                for par, _ in mut_clf.tune_priors:
                    out_pars[mtype][ex_lbl][par] = clf_params[par]

                out_time[mtype][ex_lbl]['avg'] = cv_output['mean_fit_time']
                out_time[mtype][ex_lbl]['std'] = cv_output['std_fit_time']
                out_acc[mtype][ex_lbl]['avg'] = cv_output['mean_test_score']
                out_acc[mtype][ex_lbl]['std'] = cv_output['std_test_score']
                out_acc[mtype][ex_lbl]['par'] = cv_output['params']

                mut_clf.fit_coh(cdata, mtype, exclude_feats=ex_genes,
                                exclude_samps=ex_samps)

                out_pred[mtype][ex_lbl] = {
                    'test': np.round(mut_clf.parse_preds(
                        mut_clf.predict_test(cdata, lbl_type='raw',
                                             exclude_feats=ex_genes)
                        ), 7)
                    }

                if ex_samps & set(cdata.get_train_samples()):
                    out_pred[mtype][ex_lbl]['train'] = np.round(
                        mut_clf.parse_preds(mut_clf.predict_train(
                            cdata, lbl_type='raw',
                            exclude_feats=ex_genes, include_samps=ex_samps
                            )),
                        7)

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

