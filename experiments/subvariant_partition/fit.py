
"""Finding the optimal partition of a gene's sub-variants.

This module contains classes and methods for finding the partition of a
gene's mutations that best clusters them according to downstream expression
effects.

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
from scipy.stats import norm
import random

import dill as pickle
import synapseclient


class PartitionOptim(object):
    """A class for finding the optimal partition of a gene's sub-types.

    Attributes:
        cdata (VariantCohort): A set of tumour samples with expression and
        variant call data, split into training and testing cohorts.
        pred_scores (pd.DataFrame)
        mtype_scores (dict)

    Examples:
        >>> PartitionOptim(cdata, MuType({('Gene', 'TP53'): None}),
                           'TP53', Lasso)
        >>> PartitionOptim(cdata, MuType({('Gene', 'CDH1'): None}),
                           'CDH1', rForest)

    """

    def __init__(self,
                 cdata, base_mtype, gene, clf,
                 use_lvls=('Form', 'Exon', 'Protein'), verbose=10):

        self.cdata = cdata
        self.pred_scores = pd.DataFrame(index=cdata.test_samps)
        self.mtype_scores = {}

        self.base_mtype = base_mtype
        self.base_train_samps = base_mtype.get_samples(cdata.train_mut)
        self.base_test_samps = base_mtype.get_samples(cdata.test_mut)

        self.gene = gene
        self.clf = clf
        self.verbose = verbose

        self.use_lvls = use_lvls
        self.lvl_index = 0

        self.cur_mtype = base_mtype
        self.next_mtypes = []
        self.back_mtypes = []
        self.best_mtypes = set()

    def __str__(self):
        return ("The current state of the optimizer is:\n"
                "\tCurrent mtype: {}\n\tCurrent level: {}".format(
                    self.cur_mtype, self.use_lvls[self.lvl_index]))

    def score_mtypes(self, mtypes):
        """Scores a list of mutation sub-types by classification accuracy."""

        out_scores = {mtype: {'All': 0, 'Null': 0} for mtype in mtypes}
        for mtype in mtypes:

            # if we have already tested this sub-type, retrieve its
            # classification performances from the optimizer's history
            if mtype in self.mtype_scores:
                out_scores[mtype]['All'] = self.mtype_scores[mtype]['All']
                out_scores[mtype]['Null'] = self.mtype_scores[mtype]['Null']

            # otherwise, get the samples that have the optimizer's base
            # mutation but not any mutations in this sub-type
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
            
            if self.verbose > 1:
                print("Testing {} :"
                      "\n\tAUC using all samples:\t\t{:.4f}"
                      "\n\tAUC w/o other mutated samples:\t{:.4f}"
                        .format(mtype,
                                out_scores[mtype]['All'],
                                out_scores[mtype]['Null']))

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
                                    random.randint(6, len(mtype_samps) - 1))
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
        mtype_proj = {mtype: 0 for mtype in mtypes}

        for mtype in mtypes:
            mtype_samps = mtype.get_samples(self.cdata.test_mut)

            left_scores = self.pred_scores.loc[
                cur_samps - mtype_samps, (mtype, )]
            null_scores = self.pred_scores.loc[
                self.cdata.test_samps - self.base_test_samps, (mtype, )]
            mtype_scores = self.pred_scores.loc[mtype_samps, (mtype, )]

            null_param = norm.fit(null_scores)
            mtype_param = norm.fit(mtype_scores)

            left_null_likl = np.sum(norm.logpdf(left_scores, *null_param))
            left_mtype_likl = np.sum(norm.logpdf(left_scores, *mtype_param))

            mtype_proj[mtype] = left_mtype_likl - left_null_likl

        return mtype_proj

    def step(self):
        """Performs a step of the mutation sub-type search tree space.
        
        Returns:
            out_scores (dict): Classification performances of the sub-types
                               tested in this step.
        
        """

        # if we are taking the first step of the optimizer, test the mutation
        # sub-type at the top of the search tree space
        if self.cur_mtype == self.base_mtype and self.lvl_index == 0:
            self.mtype_scores.update(self.score_mtypes([self.base_mtype]))

        # find the sub-types that are the children of the current sub-type
        # in the search space
        test_mtypes = self.cdata.train_mut.combtypes(
            mtype=self.cur_mtype, sub_levels=[self.use_lvls[self.lvl_index]],
            comb_sizes=(1, 2, 3), min_type_size=10
            )

        # filters out sub-types that are similar enough to the current type
        # that they have the same sample set in the testing cohort
        test_mtypes = [
            mtype for mtype in test_mtypes
            if (mtype.get_samples(self.cdata.test_mut)
                != self.cur_mtype.get_samples(self.cdata.test_mut))
            ]

        if self.verbose > 0:
            if len(test_mtypes) == 0:
                print("\nDid not find any sub-types to test.")
            elif len(test_mtypes) == 1:
                print("\nFound one sub-type to test...")
            else:
                print("\nFound {} sub-types to test...".format(
                    len(test_mtypes)))

        # test the performance of the children sub-types, update the
        # search history of the optimizer, add the performance of the
        # current sub-type to the returned scores for comparison
        out_scores = self.score_mtypes(test_mtypes)
        self.mtype_scores.update(out_scores)
        out_scores = (
            list(out_scores.items())
            + [(self.cur_mtype, self.mtype_scores[self.cur_mtype])]
            )
        
        return out_scores

    def traverse_branch(self):
        """Traverses a branch of the mutation sub-type search tree.

        Returns:
            status (bool): Whether or not we have exhausted the search space.

        """

        if self.verbose > 0:
            print('\n-------------')
            print(self)
            print('-------------')

        # removes the top member of the back mutation sub-types if it's empty
        if self.back_mtypes and not self.back_mtypes[-1][0]:
            self.back_mtypes = self.back_mtypes[:-1]

        null_scores = self.get_null_performance()
        new_scores = self.step()

        # if we have found sub-types of the current mutation type...
        if len(new_scores) > 1:

            # ...finds how functionally similar the classification signatures
            # of the sub-types are to the signature of the current type
            mtype_proj = self.get_clf_projection(
                [mtype for mtype, _ in new_scores[:-1]])

            if self.verbose > 1:
                print("\nRemaining samples' probability of being labelled "
                      "positively with each sub-type's classifier:")
                print('\n'.join(
                    '\t{}:\n\t\t{:.1%}'.format(s[0], (1 + 2 ** -s[1]) ** -1)
                    for s in mtype_proj.items()))

            mtype_sizes = [len(mtype.get_samples(self.cdata.train_mut))
                           for mtype, _ in new_scores]

            null_wghts = [
                [exp(-(((nl_sz - new_sz) ** 2) / new_sz ** 2)) ** 2
                 for nl_sz, _ in null_scores]
                for (_, acc), new_sz in zip(new_scores, mtype_sizes)
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

            if self.verbose > 2:
                print("\nResults of background classification analysis "
                      "(mean +/- sd) :")

                print('\n'.join(
                    '\t{}:\n\t\t{:.4f} +/- {:.4f}'.format(
                        s[0][0], s[1], s[2])
                    for s in zip(new_scores, null_means, null_sds)
                    ))

            next_mtypes = sorted(
                [(mtype, (acc['Null'] - nl_mn) / nl_sd)
                 for (mtype, acc), nl_mn, nl_sd in
                 zip(new_scores, null_means, null_sds)],
                key=lambda x: x[1]
                )

            if self.verbose > 1:
                print("\nSub-type classification performance z-score "
                      "relative to background sample subset distribution:")
                print('\n'.join('\t{}:\n\t\t{:+.2f}'.format(*s)
                                for s in next_mtypes))

            # filters out sub-types whose classification performance is too
            # low compared to the corresponding null performance distribution
            # and whose classification signatures are too similar to that
            # of the current type
            next_mtypes = [
                x[0] for x in next_mtypes
                if (x[0] != self.cur_mtype
                    #and x[1] > max(1, dict(next_mtypes)[self.cur_mtype])
                    #and mtype_proj[x[0]] < 0)
                    and (x[1] > max(2, dict(next_mtypes)[self.cur_mtype] + 1)
                         or (mtype_proj[x[0]] < 0 and x[1] > 1)))
                ]

            # ...and at least one of these sub-types are significantly
            # better than the current type...
            if next_mtypes:
                if self.verbose > 1:
                    print("\nFound {} further sub-types...".format(
                        len(next_mtypes)))

                # ...and if we have reached the bottom of the mutation
                # annotation hierarchy, add these sub-types to the list of
                # optimal sub-types...
                if self.lvl_index == (len(self.use_lvls) - 1):
                    if self.verbose > 1:
                        print("\nFinal mutation annotation level reached, "
                              "adding following sub-types to optimal set:")
                        print('\n'.join('\t{}'.format(mtype)
                                        for mtype in next_mtypes))

                    self.best_mtypes |= set(next_mtypes)

                    # ...and then go back and interrogate previously found
                    # sub-types that were better than their parent type
                    if self.back_mtypes:
                        if self.verbose > 1:
                            print("\nGoing back up to previous sub-types...")
                        self.cur_mtype = self.back_mtypes[-1][0].pop()
                        self.lvl_index = self.back_mtypes[-1][1]

                        return True

                # ...if we can go deeper in the hierarchy, proceed to test
                # the first of these better sub-types, saving the rest for
                # later
                else:
                    self.cur_mtype = next_mtypes.pop()
                    self.lvl_index += 1
                    self.back_mtypes += [[next_mtypes, self.lvl_index]]

                    return True

        # ...otherwise check if we can skip this mutation annotation level
        # and go one level deeper to find more sub-types
        if self.lvl_index < (len(self.use_lvls) - 1):
            if self.verbose > 1:
                print("\nSkipping this mutation annotation level...")
            self.lvl_index += 1
            return True

        # ...otherwise check to see if there are sub-types at the current
        # mutation level we can go back to...
        elif self.back_mtypes:
            if self.verbose > 1:
                print("\nAdding the current sub-type to the optimal set.")
                print("\nGoing back up to previous sub-types...")

            self.best_mtypes |= set(self.cur_mtype)
            self.cur_mtype = self.back_mtypes[-1][0].pop()
            self.lvl_index = self.back_mtypes[-1][1]

            return True

        # ...otherwise the traversal of this tree is complete
        else:
            return False


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
                          syn=syn, cv_seed=(int(argv[3]) + 3) * 17)

    base_mtype = MuType({('Gene', argv[1]): None})
    optim = PartitionOptim(cdata, base_mtype, argv[1], eval(argv[2]),
                           ('Form', 'Exon', 'Protein'))

    while optim.traverse_branch():
        pass

    # saves classifier results to file
    out_file = os.path.join(
        out_dir, 'results', 'out__cv-{}.p'.format(argv[3]))
    pickle.dump({'best': optim.best_mtypes, 'hist': optim.mtype_scores},
                open(out_file, 'wb'))


if __name__ == "__main__":
    main(sys.argv[1:])

