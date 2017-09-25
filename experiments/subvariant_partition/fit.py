
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
import pandas as pd

from math import exp
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import norm

import random
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
        self.pred_scores = pd.DataFrame(index=cdata.test_samps)

        self.base_train_samps = base_mtype.get_samples(cdata.train_mut)
        self.base_test_samps = base_mtype.get_samples(cdata.test_mut)

        self.use_lvls = use_lvls
        self.lvl_index = 0

        self.mtype_scores = {}
        self.cur_mtype = base_mtype
        self.next_mtypes = []
        self.back_mtypes = []
        self.best_mtypes = []

    def __str__(self):
        return ("Current mtype: {}\n"
                "Current level: {}".format(
                    self.cur_mtype, self.use_lvls[self.lvl_index]))

    def score_mtypes(self, mtypes, verbose=False):
        """Scores a list of mutation sub-types by classification accuracy."""

        out_scores = {mtype: {'All': 0, 'Null': 0} for mtype in mtypes}
        for mtype in mtypes:

            if mtype in self.mtype_scores:
                out_scores[mtype]['All'] = self.mtype_scores[mtype]['All']
                out_scores[mtype]['Null'] = self.mtype_scores[mtype]['Null']

            else:
                ex_train = (self.base_train_samps
                            - mtype.get_samples(self.cdata.train_mut))
                ex_test = (self.base_test_samps
                           - mtype.get_samples(self.cdata.test_mut))

                use_clf = self.clf()
                use_clf.tune_coh(self.cdata, mtype, tune_splits=4,
                                 test_count=12, parallel_jobs=12,
                                 exclude_genes=[self.gene])

                use_clf.fit_coh(self.cdata, mtype, exclude_genes=[self.gene])
                out_scores[mtype]['All'] = use_clf.eval_coh(
                    self.cdata, mtype, exclude_genes=[self.gene])

                if mtype != self.base_mtype:
                    use_clf = self.clf()

                    use_clf.tune_coh(
                        self.cdata, mtype, tune_splits=8,
                        test_count=24, parallel_jobs=12,
                        exclude_genes=[self.gene], exclude_samps=ex_train
                        )

                    use_clf.fit_coh(self.cdata, mtype,
                                    exclude_genes=[self.gene],
                                    exclude_samps=ex_train)

                    out_scores[mtype]['Null'] = use_clf.eval_coh(
                        self.cdata, mtype,
                        exclude_genes=[self.gene], exclude_samps=ex_test
                        )


                else:
                    out_scores[mtype]['Null'] = out_scores[mtype]['All']
                
                self.pred_scores[mtype] = [
                    x[0] for x in use_clf.predict_omic(
                        self.cdata.omic_mat.loc[
                            self.pred_scores.index,
                            self.cdata.subset_genes(
                                exclude_genes=[self.gene])
                            ]
                        )
                    ]
            
            if verbose:
                print(mtype)
                print(round(out_scores[mtype]['All'], 4))
                print(round(out_scores[mtype]['Null'], 4))

        return out_scores

    def get_null_performance(self, draw_count=50):
        """Estimates the null score distribution for the current sub-type.

        Args:
            draw_count (int): How many random subsets of samples to test.

        Examples:
            >>> optim.get_null_performance(5)
            >>> optim.get_null_performance(100)

        """

        mtype_samps = self.cur_mtype.get_samples(self.cdata.train_mut)
        draw_perfs = []

        samp_draws = [random.sample(mtype_samps,
                                    random.randint(10, len(mtype_samps) - 1))
                      for _ in range(draw_count)]

        for samp_draw in samp_draws:
            use_clf = self.clf()
            ex_samps = mtype_samps - set(samp_draw)

            use_clf.tune_coh(self.cdata, self.cur_mtype,
                             tune_splits=8, test_count=24, parallel_jobs=12,
                             exclude_genes=[self.gene],
                             exclude_samps=ex_samps)
            
            use_clf.fit_coh(self.cdata, self.cur_mtype,
                            exclude_genes=[self.gene],
                            exclude_samps=ex_samps)

            draw_perfs += [[
                len(samp_draw),
                use_clf.eval_coh(self.cdata, self.cur_mtype,
                                 exclude_genes=[self.gene],
                                 exclude_samps=ex_samps)
                ]]

        return draw_perfs

    def get_clf_projection(self, mtypes):
        cur_samps = self.cur_mtype.get_samples(self.cdata.test_mut)
        mtype_proj = {mtype: -1 for mtype in mtypes}

        for mtype in mtypes:
            mtype_samps = mtype.get_samples(self.cdata.test_mut)

            if mtype_samps != cur_samps:
                left_scores = self.pred_scores.loc[
                    cur_samps - mtype_samps, (mtype, )]
                null_scores = self.pred_scores.loc[
                    self.cdata.test_samps - self.base_test_samps, (mtype, )]
                mtype_scores = self.pred_scores.loc[mtype_samps, (mtype, )]

                null_param = norm.fit(null_scores)
                mtype_param = norm.fit(mtype_scores)

                left_null_likl = np.sum(
                    norm.logpdf(left_scores, *null_param))
                left_mtype_likl = np.sum(
                    norm.logpdf(left_scores, *mtype_param))

                mtype_proj[mtype] = left_mtype_likl - left_null_likl

        return mtype_proj

    def step(self, verbose=False):
        if self.cur_mtype == self.base_mtype:
            self.mtype_scores.update(self.score_mtypes(
                [self.base_mtype], verbose))

        test_mtypes = self.cdata.train_mut.combtypes(
            mtype=self.cur_mtype, sub_levels=[self.use_lvls[self.lvl_index]],
            comb_sizes=(1, 2, 3), min_type_size=10
            )

        out_scores = self.score_mtypes(test_mtypes, verbose)
        self.mtype_scores.update(out_scores)

        return out_scores

    def traverse_branch(self, verbose=False):

        if verbose:
            print(self)

        if self.back_mtypes and not self.back_mtypes[-1][0]:
            self.back_mtypes = self.back_mtypes[:-1]

        null_scores = self.get_null_performance()
        new_scores = self.step(verbose)

        # if we have found sub-types of the current mutation type...
        if new_scores:

            mtype_proj = self.get_clf_projection(
                [mtype for mtype, _ in new_scores.items()])
            print(mtype_proj)

            mtype_sizes = [len(mtype.get_samples(self.cdata.train_mut))
                           for mtype, _ in new_scores.items()]


            null_wghts = [
                [exp(-(((nl_sz - new_sz) ** 2) / new_sz ** 2)) ** 2
                 for nl_sz, _ in null_scores]
                for (_, acc), new_sz in zip(new_scores.items(), mtype_sizes)
                ]

            wght_sums = [sum(wghts) for wghts in null_wghts]
            null_wghts = [[wght / wght_sum for wght in wghts]
                          for wghts, wght_sum in zip(null_wghts, wght_sums)]

            null_means = [sum(wght * perf
                              for wght, (_, perf) in zip(wghts, null_scores))
                          for wghts in null_wghts]

            null_crcts = [sum(wght ** 2 for wght in wghts)
                          for wghts in null_wghts]

            null_sds = [
                (sum(wght * (perf - nl_mn) ** 2
                     for wght, (_, perf) in zip(wghts, null_scores))
                 / (1 - null_crct)) ** 0.5
                for wghts, nl_mn, null_crct in
                zip(null_wghts, null_means, null_crcts)
                ]

            next_mtypes = sorted(
                [(mtype, (acc['Null'] - nl_mn) / nl_sd)
                 for (mtype, acc), nl_mn, nl_sd in
                 zip(new_scores.items(), null_means, null_sds)],
                key=lambda x: x[1]
                )

            next_mtypes = [x[0] for x in next_mtypes
                           if x[1] > 2 and mtype_proj[x[0]] < 0]
            print(next_mtypes)

            # ...and at least one of these sub-types are significantly
            # better than the current type...
            if next_mtypes:

                # ...and if we have reached the bottom of the mutation
                # annotation hierarchy, add these sub-types to the list of
                # optimal sub-types...
                if self.lvl_index == (len(self.use_lvls) - 1):
                    self.best_mtypes += next_mtypes

                    # ...and then go back and interrogate previously found
                    # sub-types that were better than their parent type
                    if self.back_mtypes:
                        self.cur_mtype = self.back_mtypes[-1][0].pop()
                        return self.traverse_branch(verbose)

                # ...if we can go deeper in the hierarchy, proceed to test
                # the first of these better sub-types, saving the rest for
                # later
                else:
                    self.cur_mtype = next_mtypes.pop()
                    self.back_mtypes += [[next_mtypes, self.lvl_index]]
                    self.lvl_index += 1

                    return self.traverse_branch(verbose)

        # ...otherwise check if we can skip this mutation annotation level
        # and go one level deeper to find more sub-types
        if self.lvl_index < (len(self.use_lvls) - 1):
            self.lvl_index += 1
            return self.traverse_branch(verbose)

        # ...otherwise check to see if there are sub-types at the current
        # mutation level we can go back to...
        if self.back_mtypes:
            self.best_mtypes.append(self.cur_mtype)
            self.cur_mtype = self.back_mtypes[-1][0].pop()
            self.lvl_index = self.back_mtypes[-1][1]

            return self.traverse_branch(verbose)


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
                          mut_levels=('Gene', 'Form', 'Exon', 'Protein'),
                          syn=syn, cv_seed=(int(argv[3]) + 3) * 19)

    base_mtype = MuType({('Gene', argv[1]): None})
    optim = PartitionOptim(cdata, base_mtype, argv[1], eval(argv[2]),
                           ('Form', 'Exon', 'Protein'))

    optim.traverse_branch(verbose=True)
    print(optim.cur_mtype)
    print(optim.best_mtypes)
    print(optim.next_mtypes)
    print(optim.back_mtypes)
    print(optim.lvl_index)
    optim.traverse_branch(verbose=True)

    # saves classifier results to file
    out_file = os.path.join(
        out_dir, 'results', 'out__cv-{}.p'.format(argv[3]))
    pickle.dump(optim, open(out_file, 'wb'))


if __name__ == "__main__":
    main(sys.argv[1:])

