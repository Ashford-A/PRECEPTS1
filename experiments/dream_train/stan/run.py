
import os
base_dir = os.path.dirname(__file__)

import sys
sys.path.extend([os.path.join(base_dir, '../../../..')])

import synapseclient
import dill as pickle

from HetMan.features.cohorts import TransferDreamCohort
import HetMan.predict.bayesian_pathway.multi_protein as mpt


def main(argv):

    # use your own Synapse cache and credentials here
    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = ("/home/exacloud/lustre1/CompBio/"
                                "mgrzad/input-data/synapse")
    syn.login()

    # loads the challenge data
    cdata = TransferDreamCohort(
        syn, argv[0], intx_types=[argv[1]],
        cv_seed=171, cv_prop=0.8, miss_cutoff=0.1
        )

    out_dir = os.path.join(base_dir, 'output')
    out_file = os.path.join(out_dir, argv[0], argv[1], 'results', 'coefs.p')

    # initializes the model and fits it using all of the genes in the
    # `inter`section of the RNA genes, CNA genes, and proteome genes
    clf = mpt.StanProteinPipe(argv[1])

    # finds the best combination of model hyper-parameters, uses these
    # parameters to fit to the data
    #clf.tune_coh(cdata, pheno='inter',
    #             tune_splits=4, test_count=4, parallel_jobs=16)
    clf.fit_coh(cdata, pheno='inter') 
    clf.eval_coh(cdata, pheno='inter')

    pickle.dump(clf, open(out_file, 'wb'))


if __name__ == "__main__":
    main(sys.argv[1:])

