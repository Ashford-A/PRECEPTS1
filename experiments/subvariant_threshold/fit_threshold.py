
import os
import sys
base_dir = os.path.dirname(__file__)
sys.path.extend([os.path.join(base_dir, '..', '..', '..')])

from HetMan.experiments.subvariant_tour.fit_tour import get_excluded_genes
from HetMan.experiments.subvariant_infer.fit_infer import transfer_model
from dryadic.learning.classifiers import *

import argparse
import dill as pickle
from pathlib import Path
from joblib import Parallel, delayed


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

    args = parser.parse_args()
    setup_dir = os.path.join(args.use_dir, 'setup')
    clf = eval(args.classif)
    mut_clf = clf()

    cdata = None
    while cdata is None:
        try:
            with open(os.path.join(setup_dir, 'cohort-data.p'),
                      'rb') as cdata_f:
                cdata = pickle.load(cdata_f)

        except:
            print("Failed to load cohort data, trying again...")
            time.sleep(61)

    with open(os.path.join(setup_dir, "feat-list.p"), 'rb') as fl:
        feat_list = pickle.load(fl)
    with open(os.path.join(setup_dir, "muts-list.p"), 'rb') as muts_f:
        mtype_list = pickle.load(muts_f)

    coh_files = Path(setup_dir).glob("*__cohort-data.p")
    coh_dict = {coh_fl.stem.split('__')[0]: coh_fl for coh_fl in coh_files}
    out_inf = {mtype: None for mtype in mtype_list}
    out_trnsf = {mtype: {coh: None for coh in coh_dict}
                 for mtype in mtype_list}

    # for each subtype, check if it has been assigned to this task
    for i, mtype in enumerate(mtype_list):
        if (i % args.task_count) == args.task_id:
            print("Testing {} ...".format(mtype))

            use_feats = feat_list - get_excluded_genes(
                'Chrm', mtype.base_mtype.get_labels()[0], cdata.gene_annot)

            # tune the hyper-parameters of the classifier
            mut_clf, _ = mut_clf.tune_coh(cdata, mtype,
                                          include_feats=use_feats,
                                          tune_splits=4, test_count=48,
                                          parallel_jobs=8)

            out_inf[mtype] = mut_clf.infer_coh(
                cdata, mtype, include_feats=use_feats,
                infer_splits=80, infer_folds=4, parallel_jobs=8
                )
            mut_clf.fit_coh(cdata, mtype, include_feats=use_feats)

            out_trnsf[mtype] = dict(zip(coh_dict.keys(), [
                mut_clf.parse_preds(vals) for vals in Parallel(
                    backend='threading', n_jobs=8, pre_dispatch=8)(
                        delayed(transfer_model)(trnsf_fl, mut_clf, use_feats)
                        for trnsf_fl in coh_dict.values())
                ]))

        else:
            del(out_inf[mtype])
            del(out_trnsf[mtype])

    with open(os.path.join(args.use_dir, 'output',
                           "out_task-{}.p".format(args.task_id)),
              'wb') as fl:
        pickle.dump({'Infer': out_inf, 'Transfer': out_trnsf,
                     'Clf': mut_clf.__class__},
                    fl, protocol=-1)


if __name__ == "__main__":
    main()

