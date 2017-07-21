
import os
base_dir = os.path.dirname(__file__)

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])

import numpy as np
import pickle

from itertools import product
from itertools import combinations as combn

from HetMan.features.variants import MuType
from HetMan.features.cohorts import VariantCohort
from HetMan.predict.classifiers import Lasso
from sklearn.metrics.pairwise import cosine_similarity

import synapseclient


def main(argv):
    """Runs the experiment."""

    print(argv)
    out_dir = os.path.join(base_dir, 'output', argv[0], argv[1])
    coh_lbl = 'TCGA-{}'.format(argv[0])
    mtype_list = pickle.load(
        open(os.path.join(out_dir, 'tmp', 'mtype_list.p'), 'rb'))

    syn = synapseclient.Synapse()
    syn.login()
    cdata = VariantCohort(syn, cohort=coh_lbl, mut_genes=[argv[1]],
                          mut_levels=['Gene', 'Form', 'Exon', 'Location'],
                          cv_seed=99)

    base_mtype = MuType({('Gene', argv[1]): None})
    tp53_train_samps = base_mtype.get_samples(cdata.train_mut)
    tp53_test_samps = base_mtype.get_samples(cdata.test_mut)

    out_stat = {mtype: None for mtype in mtype_list}
    out_coef = {mtype: None for mtype in mtype_list}
    out_acc = {mtype: None for mtype in mtype_list}
    out_pred = {mtype: None for mtype in mtype_list}

    out_cross = {mtypes: [None, None, None, None]
                 for mtypes in product(mtype_list, mtype_list)}
    out_mutex = {tuple(sorted(mtypes)): None
                 for mtypes in combn(mtype_list, 2)}

    for i, mtype in enumerate(mtype_list):
        if i % 10 == (int(argv[-1]) - 1):
            print(mtype)

            ex_train = tp53_train_samps - mtype.get_samples(cdata.train_mut)
            ex_test = tp53_test_samps - mtype.get_samples(cdata.test_mut)

            test_stat = cdata.test_pheno(mtype)
            out_stat[mtype] = np.where(test_stat)
            print(np.sum(test_stat))

            clf = Lasso()
            clf.tune_coh(cdata, mtype, tune_splits=4,
                         test_count=16, parallel_jobs=8,
                         exclude_genes=[argv[1]], exclude_samps=ex_train)
            print(clf)

            clf.fit_coh(cdata, mtype,
                        exclude_genes=[argv[1]], exclude_samps=ex_train)
            out_coef[mtype] = {gene: val for gene, val in
                               clf.get_coef().items() if val != 0}

            out_acc[mtype] = clf.eval_coh(
                cdata, mtype, exclude_genes=[argv[1]], exclude_samps=ex_test)
            out_pred[mtype] = np.array(
                clf.predict_test(cdata, exclude_genes=[argv[1]]))

            for other_mtype in mtype_list:
                other_stat = cdata.test_pheno(other_mtype)

                none_which = ~test_stat & ~other_stat
                if np.sum(none_which) >= 5:
                    out_cross[mtype, other_mtype][0] = np.mean(
                        out_pred[mtype][none_which])

                other_which = ~test_stat & other_stat
                if np.sum(other_which) >= 5:
                    out_cross[mtype, other_mtype][1] = np.mean(
                        out_pred[mtype][other_which])

                cur_which = test_stat & ~other_stat
                if np.sum(cur_which) >= 5:
                    out_cross[mtype, other_mtype][2] = np.mean(
                        out_pred[mtype][cur_which])

                both_which = test_stat & other_stat
                if np.sum(both_which) >= 5:
                    out_cross[mtype, other_mtype][3] = np.mean(
                        out_pred[mtype][both_which])

            for mtypes in list(out_mutex.keys()):
                if mtypes[0] == mtype:
                    out_mutex[mtypes] = cdata.mutex_test(mtypes[0], mtypes[1])

        else:
            del(out_stat[mtype])
            del(out_coef[mtype])
            del(out_acc[mtype])
            del(out_pred[mtype])

            for other_mtype in mtype_list:
                del(out_cross[mtype, other_mtype])
            
            for mtypes in list(out_mutex.keys()):
                if mtypes[0] == mtype:
                    del(out_mutex[mtypes])

    # saves classifier results to file
    out_file = os.path.join(out_dir, 'results', 'ex___run' + argv[-1] + '.p')
    pickle.dump({'Stat': out_stat, 'Coef': out_coef, 'Mutex': out_mutex,
                 'Acc': out_acc, 'Pred': out_pred, 'Cross': out_cross},
                open(out_file, 'wb'))


if __name__ == "__main__":
    main(sys.argv[1:])

