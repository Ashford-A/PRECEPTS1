
import os
import sys
base_dir = os.path.dirname(__file__)
sys.path.extend([os.path.join(base_dir, '../../..')])

from dryadic.learning.classifiers import *
import argparse
import dill as pickle


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('classif', type=str,
                        help='the name of a mutation classifier')
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

    with open(os.path.join(setup_dir, "pairs-list.p"), 'rb') as muts_f:
        pairs_list = pickle.load(muts_f)

    clf = eval(args.classif)
    clf.predict_proba = clf.calc_pred_labels
    mut_clf = clf()
 
    out_tune = {mtypes: [{par: None for par, _ in mut_clf.tune_priors}] * 2
                for mtypes in pairs_list}
    out_inf = {mtypes: [None, None] for mtypes in pairs_list}

    for i, (mtype1, mtype2) in enumerate(pairs_list):
        if (i % args.task_count) == args.task_id:
            print("Testing {} x {} ...".format(mtype1, mtype2))

            use_gene1 = mtype1.subtype_list()[0][0]
            use_gene2 = mtype2.subtype_list()[0][0]
            use_samps1 = mtype1.get_samples(cdata.mtree)
            use_samps2 = mtype2.get_samples(cdata.mtree)

            ex_genes = {
                gene for gene, annot in cdata.gene_annot.items()
                if annot['Chr'] in {cdata.gene_annot[use_gene1]['Chr'],
                                    cdata.gene_annot[use_gene2]['Chr']}
                }

            mut_clf.tune_coh(cdata, mtype1, exclude_feats=ex_genes,
                             exclude_samps=use_samps2, tune_splits=4,
                             test_count=24, parallel_jobs=8)

            clf_params = mut_clf.get_params()
            for par, _ in mut_clf.tune_priors:
                out_tune[(mtype1, mtype2)][0][par] = clf_params[par]

            out_inf[(mtype1, mtype2)][0] = mut_clf.infer_coh(
                cdata, mtype1,
                exclude_feats=ex_genes, force_test_samps=use_samps2,
                infer_splits=40, infer_folds=4, parallel_jobs=8
                )

            mut_clf.tune_coh(cdata, mtype2, exclude_feats=ex_genes,
                             exclude_samps=use_samps1, tune_splits=4,
                             test_count=24, parallel_jobs=8)

            clf_params = mut_clf.get_params()
            for par, _ in mut_clf.tune_priors:
                out_tune[(mtype1, mtype2)][1][par] = clf_params[par]

            out_inf[(mtype1, mtype2)][1] = mut_clf.infer_coh(
                cdata, mtype2,
                exclude_feats=ex_genes, force_test_samps=use_samps1,
                infer_splits=40, infer_folds=4, parallel_jobs=8
                )

        else:
            del(out_tune[(mtype1, mtype2)])
            del(out_inf[(mtype1, mtype2)])

    with open(os.path.join(args.use_dir, 'output',
                           "out_task-{}.p".format(args.task_id)), 'wb') as f:
        pickle.dump({'Infer': out_inf, 'Tune': out_tune,
                     'Info': {'Clf': mut_clf.__class__,
                              'TunePriors': mut_clf.tune_priors}}, f)


if __name__ == "__main__":
    main()

