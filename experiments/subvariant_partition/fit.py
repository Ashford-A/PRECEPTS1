
"""Finding the optimal partition of a gene's sub-variants.

Args:
    fit.py <cohort> <gene> <classif> <cv_id>

Examples:
    fit.py BRCA TP53 Lasso 2
    fit.py UCEC PTEN ElasticNet 0
    fit.py SKCM TTN Ridge 3

"""

import os
base_dir = os.path.dirname(__file__)

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.features.variants import MuType
from HetMan.features.cohorts import VariantCohort
from HetMan.predict.classifiers import Lasso, ElasticNet, Ridge

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from itertools import product
from itertools import combinations as combn

import dill as pickle
import synapseclient


class PartitionOptim(object):

    def __init__(self,
                 cdata, base_mtype, gene, clf,
                 use_lvls=('Form', 'Exon', 'Protein')):

        self.cdata = cdata
        self.base_mtype = base_mtype
        self.gene = gene
        self.clf = clf

        self.base_train_samps = base_mtype.get_samples(cdata.train_mut)
        self.base_test_samps = base_mtype.get_samples(cdata.test_mut)

        self.use_lvls = use_lvls
        self.cur_lvl = use_lvls[0]

        self.mtype_scores = {base_mtype: {'All': 0, 'Null': 0}}
        self.cur_mtype = base_mtype
        self.next_mtypes = []
        self.back_mtypes = []
        self.best_mtypes = []

    def __str__(self):
        return ("Current mtype: {}\n"
                "Current level: {}".format(
                    self.cur_mtype, self.cur_lvl))

    def score_mtypes(self, mtypes, verbose=False):
        out_scores = {mtype: {'All': 0, 'Null': 0} for mtype in mtypes}

        for mtype in mtypes:
            ex_train = (self.base_train_samps
                        - mtype.get_samples(self.cdata.train_mut))
            ex_test = (self.base_test_samps
                       - mtype.get_samples(self.cdata.test_mut))

            use_clf = self.clf()
            use_clf.tune_coh(self.cdata, mtype, tune_splits=8,
                             test_count=24, parallel_jobs=12)
            use_clf.fit_coh(self.cdata, mtype)
            out_scores[mtype]['All'] = use_clf.eval_coh(self.cdata, mtype)

            if mtype != self.base_mtype:
                use_clf = self.clf()
                use_clf.tune_coh(self.cdata, mtype, tune_splits=8,
                                 test_count=24, parallel_jobs=12,
                                 exclude_genes=[self.gene], exclude_samps=ex_train)
                use_clf.fit_coh(self.cdata, mtype,
                                exclude_genes=[self.gene], exclude_samps=ex_train)
                out_scores[mtype]['Null'] = use_clf.eval_coh(
                    self.cdata, mtype,
                    exclude_genes=[self.gene], exclude_samps=ex_test
                    )

            else:
                out_scores[mtype]['Null'] = out_scores[mtype]['All']
            
            if verbose:
                print(mtype)
                print(round(out_scores[mtype]['All'], 4))
                print(round(out_scores[mtype]['Null'], 4))

        return out_scores

    def step(self, verbose=False):
        if self.cur_mtype == self.base_mtype:
            self.mtype_scores.update(self.score_mtypes(
                [self.base_mtype], verbose))

        test_mtypes = self.cdata.train_mut.combtypes(
            mtype=self.cur_mtype, sub_levels=[self.cur_lvl],
            comb_sizes=(1, 2, 3), min_type_size=12
            )

        out_scores = self.score_mtypes(test_mtypes, verbose)
        self.mtype_scores.update(out_scores)

        return out_scores

    def traverse_branch(self, back_mtypes=None, verbose=False):

        if verbose:
            print(self)

        new_scores = self.step(verbose)
        if new_scores:

            next_mtypes = sorted(
                [(mtype, acc['Null']) for mtype, acc in new_scores.items()
                if acc['Null'] > self.mtype_scores[self.cur_mtype]['Null']],
                key=lambda x: x[1]
                )
            next_mtypes = [x[0] for x in next_mtypes]

            if next_mtypes:
                if self.cur_lvl == self.use_lvls[-1]:
                    self.best_mtypes += next_mtypes

                else:
                    self.cur_mtype = next_mtypes.pop()
                    self.cur_lvl = self.use_lvls[
                        self.use_lvls.index(self.cur_lvl) + 1]

                    if back_mtypes is not None:
                        self.back_mtypes += [[back_mtypes]]

                    return self.traverse_branch(next_mtypes, verbose)

            elif back_mtypes:
                self.cur_mtype = back_mtypes.pop()
                self.cur_lvl = self.use_lvls[
                    self.use_lvls.index(self.cur_lvl) - 1]
                return self.traverse_branch(back_mtypes, verbose)

            elif self.cur_lvl != self.use_lvls[-1]:
                self.cur_lvl = self.use_lvls[
                    self.use_lvls.index(self.cur_lvl) + 1]
                return self.traverse_branch(verbose=verbose)

        else:
            self.best_mtypes.append(self.cur_mtype)

            if back_mtypes is not None and back_mtypes:
                self.cur_mtype = back_mtypes.pop()
                return self.traverse_branch(back_mtypes, verbose)

            elif self.back_mtypes:
                back_mtypes = self.back_mtypes.pop()
                self.cur_mtype = back_mtypes.pop()
                return self.traverse_branch(back_mtypes, verbose)


def main(argv):
    """Runs the experiment."""

    # gets the directory where output will be saved and the name of the TCGA
    # cohort under consideration, loads the list of gene sub-variants 
    print(argv)
    out_dir = os.path.join(base_dir, 'output', argv[0], argv[1], argv[2])
    coh_lbl = 'TCGA-{}'.format(argv[0])

    # loads the expression data and gene mutation data for the given TCGA
    # cohort, with the training/testing cohort split defined by the
    # cross-validation id for this task
    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = ("/home/exacloud/lustre1/CompBio/"
                                "mgrzad/input-data/synapse")
    syn.login()

    cdata = VariantCohort(cohort=coh_lbl, mut_genes=[argv[1]],
                          mut_levels=('Gene', 'Form_base', 'Exon', 'Protein'),
                          syn=syn, cv_seed=(int(argv[3]) + 3) * 19)

    base_mtype = MuType({('Gene', argv[1]): None})
    optim = PartitionOptim(cdata, base_mtype, argv[1], eval(argv[2]),
                           ('Form_base', 'Exon', 'Protein'))

    optim.traverse_branch(verbose=True)
    print(optim.cur_mtype)
    print(optim.best_mtypes)
    print(optim.next_mtypes)
    print(optim.back_mtypes)
    print(optim.cur_lvl)
    optim.traverse_branch(verbose=True)

    # saves classifier results to file
    out_file = os.path.join(
        out_dir, 'results', 'out__cv-{}.p'.format(argv[3]))
    pickle.dump(optim, open(out_file, 'wb'))


if __name__ == "__main__":
    main(sys.argv[1:])

