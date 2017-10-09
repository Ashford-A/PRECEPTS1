
import os
base_dir = os.path.dirname(__file__)

import sys
sys.path.extend([os.path.join(base_dir, '../../../..')])

import numpy as np
import pandas as pd

import synapseclient
import dill as pickle

from HetMan.features.cohorts import DreamCohort
from HetMan.experiments.dream_train.utils import *
from HetMan.experiments.dream_train.baseline.utils import load_output

from itertools import product

use_regrs = ['GradBoost', 'rForest'][::-1]
use_input = ['rna+cna', 'rna', 'cna']
out_dir = os.path.join(base_dir, 'ensemble', 'output')


def main(argv):
    """Runs the experiment."""

    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = ("/home/exacloud/lustre1/CompBio/"
                                "mgrzad/input-data/synapse")
    syn.login()

    out_data = pd.DataFrame(
        {(clf, inpt): np.min(load_output('OV', inpt, clf))
         for clf, inpt in product(use_regrs, use_input)}
        )

    for inpt in use_input:
        cdata = DreamCohort(syn, cohort='OV', omic_type=inpt, cv_prop=1.0)
        
        for clf in use_regrs:
            regr_genes = out_data.index[
                (out_data[clf][inpt]
                 > out_data.loc[:, ~out_data.columns.isin([(clf, inpt)])]
                        .max(axis=1))
            & (out_data[clf][inpt] > 0.5)
            ]

            for regr_gene in regr_genes:
                regr_obj = eval(clf)()
                gene_lbl = regr_gene.split('__')[-1]

                regr_obj.tune_coh(
                    cdata, regr_gene,
                    tune_splits=8, test_count=24, parallel_jobs=16
                    )

                regr_obj.fit_coh(cdata, regr_gene)
                regr_score = regr_obj.score_coh(cdata, regr_gene)

                if regr_score > 0.4:
                    print("Regressor {} for gene {} with input {} passed "
                          "validation with a score of {:.4f}".format(
                            clf, gene_lbl, inpt, regr_score))
            
                    out_file = os.path.join(
                        out_dir, 'regressors',
                        'fit-regr__{}__{}__{}.p'.format(gene_lbl, clf, inpt)
                        )
                    pickle.dump(regr_obj, open(out_file, 'wb'))

                else:
                    print("Regressor {} for gene {} with input {} failed "
                          "validation with a score of {:.4f}".format(
                              clf, gene_lbl, inpt, regr_score))


if __name__ == "__main__":
    main(sys.argv[1:])

