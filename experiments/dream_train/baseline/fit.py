
"""Baseline prediction for the NCI-CPTAC Dream Proteogenomic Subchallenge #2.

Args:
    python fit.py <cohort> <input> <classif> <cv_id> <task_id>

Examples:
    python fit.py BRCA rna ElasticNet 0 0
    python fit.py OV cna SVRrbf 2 11
    python fit.py BRCA rna+cna rForest 4 19

"""

import os
base_dir = os.path.dirname(__file__)

import sys
sys.path.extend([os.path.join(base_dir, '../../../..')])

from HetMan.features.cohorts import DreamCohort
from HetMan.predict.pipelines import ProteinPipe
import HetMan.predict.regressors as regr

import dill as pickle
import synapseclient


# machine learning pipelines for predicting proteomic levels from -omics data
class ElasticNet(regr.ElasticNet, ProteinPipe):
    pass

class SVRrbf(regr.SVRrbf, ProteinPipe):
    pass

class rForest(regr.rForest, ProteinPipe):
    pass

class kNeigh(regr.KNeighbors, ProteinPipe):
    pass

class GaussProc(regr.GaussianProcess, ProteinPipe):
    pass

class GradBoost(regr.GradientBoosting, ProteinPipe):
    pass


def main(argv):
    """Runs the experiment."""

    # creates the directory where the results will be stored, loads the list
    # of genes to predict the proteomic levels of and the learning pipeline
    out_dir = os.path.join(base_dir, 'output', argv[0], argv[1], argv[2])
    gene_list = pickle.load(
        open(os.path.join(out_dir, 'tmp', 'gene_list.p'), 'rb'))
    regr_cls = eval(argv[2])

    # gets a Synapse client instance, points to the directory where the
    # downloaded datasets and login credentials are to be stored
    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = ("/home/exacloud/lustre1/CompBio/"
                                "mgrzad/input-data/synapse")

    # logs into Synapse, loads the challenge data with a training/testing
    # cohort split defined by the cross-validation ID
    syn.login()
    cdata = DreamCohort(syn, cohort=argv[0], omic_type=argv[1],
                        cv_seed=(int(argv[3]) + 3) * 37)

    # for each gene whose proteomic levels are to be predicted, check if it
    # is to be tested here according to the task ID
    out_rval = {gene: 0 for gene in gene_list}
    for i, gene in enumerate(gene_list):
        if i % 20 == int(argv[4]):

            print(gene)
            regr_obj = regr_cls()

            # tunes the regressor hyper-parameters using sub-splits of the
            # training cohort, fits it using the entire training cohort
            regr_obj.tune_coh(cdata, gene,
                              tune_splits=4, test_count=32, parallel_jobs=8)
            regr_obj.fit_coh(cdata, gene)

            # score the performance of the regressor for this gene using the
            # testing cohort, show the tuned hyper-parameters
            print(regr_obj)
            out_rval[gene] = regr_obj.eval_coh(cdata, gene)
            print(out_rval[gene])

        else:
            del(out_rval[gene])

    # saves classifier results to file
    out_file = os.path.join(out_dir, 'results',
                            'out__cv-{}_task-{}.p'.format(argv[3], argv[4]))
    pickle.dump(out_rval, open(out_file, 'wb'))


if __name__ == "__main__":
    main(sys.argv[1:])

