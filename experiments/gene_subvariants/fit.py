
"""Finding the downstream expression effect of gene sub-variants.

This script tries to identify the expression signature of a list of
sub-variants. A sub-variant can be any subset of the given gene's mutations,
but is here limited to groups of mutations defined by common form (i.e.
splice site mutations, missense mutations, frameshift mutations) or location
(i.e. 5th exon, 123rd protein).

To allow for parallelization, we split the list of sub-variants into equally
sized tasks that are each tested in a separate cluster array job. The split
is done by taking the modulus of each sub-variant's position in the master
list of sub-variants, which has been created by the setup.py script. We
repeat this process for multiple splits of the TCGA cohort into training/
testing cohorts, as defined by the cross-validation ID.

Args:
    fit.py <cohort> <gene> <cv_id> <task_id>

Examples:
    fit.py BRCA TP53 2 4
    fit.py UCEC PTEN 0 2
    fit.py SKCM TTN 3 1

"""

import os
base_dir = os.path.dirname(__file__)

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.features.variants import MuType
from HetMan.features.cohorts import VariantCohort
from HetMan.predict.classifiers import Lasso

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from itertools import product
from itertools import combinations as combn

import pickle
import synapseclient


def main(argv):
    """Runs the experiment."""

    # gets the directory where output will be saved and the name of the TCGA
    # cohort under consideration, loads the list of gene sub-variants 
    print(argv)
    out_dir = os.path.join(base_dir, 'output', argv[0], argv[1])
    coh_lbl = 'TCGA-{}'.format(argv[0])
    mtype_list = pickle.load(
        open(os.path.join(out_dir, 'tmp', 'mtype_list.p'), 'rb'))

    # loads the expression data and gene mutation data for the given TCGA
    # cohort, with the training/testing cohort split defined by the
    # cross-validation id for this task
    syn = synapseclient.Synapse()
    syn.login()
    cdata = VariantCohort(
        syn, cohort=coh_lbl, mut_genes=[argv[1]],
        mut_levels=['Gene', 'Form_base', 'Exon', 'Location'],
        cv_seed=(int(argv[2]) + 3) * 19
        )

    # gets the mutation type representing all of the mutations for the given
    # gene, finds which samples have these mutations in the training and
    # testing cohorts
    base_mtype = MuType({('Gene', argv[1]): None})
    tp53_train_samps = base_mtype.get_samples(cdata.train_mut)
    tp53_test_samps = base_mtype.get_samples(cdata.test_mut)

    out_stat = {mtype: None for mtype in mtype_list}
    out_coef = {mtype: None for mtype in mtype_list}
    out_acc = {mtype: None for mtype in mtype_list}
    out_pred = {mtype: None for mtype in mtype_list}

    # for each of the gene's sub-variants, check if it has been assigned to
    # this task
    for i, mtype in enumerate(mtype_list):
        if i % 8 == int(argv[3]):
            print(mtype)

            ex_train = tp53_train_samps - mtype.get_samples(cdata.train_mut)
            ex_test = tp53_test_samps - mtype.get_samples(cdata.test_mut)

            out_stat[mtype] = np.where(cdata.test_pheno(mtype))[0].tolist()
            print(len(out_stat[mtype]))

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
            out_pred[mtype] = clf.predict_test(cdata, exclude_genes=[argv[1]])

        else:
            del(out_stat[mtype])
            del(out_coef[mtype])
            del(out_acc[mtype])
            del(out_pred[mtype])

    # saves classifier results to file
    out_file = os.path.join(out_dir, 'results',
                            'out__cv-{}_task-{}.p'.format(argv[2], argv[3]))
    pickle.dump({'Stat': out_stat, 'Coef': out_coef,
                 'Acc': out_acc, 'Pred': out_pred},
                open(out_file, 'wb'))


if __name__ == "__main__":
    main(sys.argv[1:])

