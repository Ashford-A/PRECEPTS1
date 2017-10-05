
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

use_regrs = ['ElasticNet', 'kNeigh', 'GradBoost', 'rForest']
out_dir = os.path.join(base_dir, 'ensemble', 'output')


def main(argv):
    """Runs the experiment."""

    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = ("/home/exacloud/lustre1/CompBio/"
                                "mgrzad/input-data/synapse")

    syn.login()
    cdata = DreamCohort(syn, cohort='OV', omic_type='rna', cv_prop=1.0)

    out_data = pd.DataFrame({clf: np.min(load_output('OV', 'rna', clf))
                             for clf in use_regrs})

    for clf in use_regrs:

        regr_genes = out_data.index[
            (out_data[clf]
             > out_data.loc[:, ~out_data.columns.isin([clf])].max(axis=1))
            & (out_data[clf] > 0.5)
            ]

        for regr_gene in regr_genes:
            regr_obj = eval(clf)()
            gene_lbl = regr_gene.split('__')[-1]

            regr_obj.tune_coh(cdata, regr_gene,
                              tune_splits=12, test_count=36, parallel_jobs=12) 
            regr_obj.fit_coh(cdata, regr_gene)

            regr_score = regr_obj.score_coh(cdata, regr_gene)
            if regr_score > 0.4:

                print("Regressor {} for gene {} passed validation "
                      "with a score of {:.4f}".format(
                          clf, gene_lbl, regr_score))
            
                out_file = os.path.join(
                    out_dir, 'regressors',
                    'fit-regr__{}__{}.p'.format(gene_lbl, clf)
                    )
                pickle.dump(regr_obj, open(out_file, 'wb'))

            else:
                print("Regressor {} for gene {} failed validation "
                      "with a score of {:.4f}".format(
                          clf, gene_lbl, regr_score))



if __name__ == "__main__":
    main(sys.argv[1:])

