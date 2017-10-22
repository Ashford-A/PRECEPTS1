
import os
base_dir = os.path.dirname(__file__)

import sys
sys.path.extend([os.path.join(base_dir, '../../../..')])

import synapseclient
import dill as pickle
from glob import glob

from HetMan.features.cohorts import TransferDreamCohort
import HetMan.predict.bayesian_pathway.multi_protein as mpt


def load_output(out_dir):

    out_files = [
        glob(os.path.join(out_dir, 'results/out_*_{}.p'.format(cv_id)))
        for cv_id in range(10)
        ]
    
    return [{fl.split('results/')[-1].split('_')[1]:
             pickle.load(open(fl, 'rb')) for fl in fls}
            for fls in out_files]

def main(argv):

    # use your own Synapse cache and credentials here
    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = ("/home/exacloud/lustre1/CompBio/"
                                "mgrzad/input-data/synapse")
    syn.login()

    # loads the challenge data
    cdata = TransferDreamCohort(syn, argv[0], intx_types=[argv[1]],
                                cv_seed=(int(argv[2]) * 41) + 1, cv_prop=0.8)

    # initializes the model and fits it using all of the genes in the
    # `inter`section of the RNA genes, CNA genes, and proteome genes
    clf = mpt.StanDefault(argv[1])

    # finds the best combination of model hyper-parameters, uses these
    # parameters to fit to the data
    #clf.tune_coh(cdata, pheno='inter',
    #             tune_splits=4, test_count=4, parallel_jobs=16)
    clf.fit_coh(cdata, pheno='inter') 

    out_file = os.path.join(base_dir, 'output', 'intx', argv[0], 'results',
                            'out_{}_{}.p'.format(argv[1], argv[2]))

    # saves the classifier performance, and the fitted posterior means of the
    # model variables and their names to file
    pickle.dump({'Eval': clf.eval_coh(cdata, pheno='inter'),
                 'PostMeans': clf.named_steps['fit'].post_means,
                 'VarNames': clf.named_steps['fit'].var_names},
                open(out_file, 'wb'))


if __name__ == "__main__":
    main(sys.argv[1:])

