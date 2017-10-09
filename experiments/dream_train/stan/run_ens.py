
import os
from glob import glob
base_dir = os.path.dirname(__file__)

import sys
sys.path.extend([os.path.join(base_dir, '../../../..')])

import synapseclient
import dill as pickle
import pandas as pd
from scipy.stats import pearsonr

from HetMan.features.cohorts import DreamCohort, TransferDreamCohort
import HetMan.predict.bayesian_pathway.multi_protein as mpt

regr_dir = '/home/exacloud/lustre1/CompBio/mgrzad/dream_regressors'


def main(argv):

    # use your own Synapse cache and credentials here
    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = ("/home/exacloud/lustre1/CompBio/"
                                "mgrzad/input-data/synapse")
    syn.login()

    # loads the challenge data
    cdata_rna = DreamCohort(syn, cohort=argv[0], omic_type='rna', cv_prop=1.0)
    known_prots = pd.DataFrame(index=cdata_rna.train_samps)

    for i, fl in enumerate(glob(os.path.join(regr_dir, "*.p"))):
        with open(fl, "rb") as handle:

            regr_gn = os.path.basename(fl).split('__')[1]
            regr_obj = pickle.load(handle)

            train_omic, train_prot = cdata_rna.train_data(
                'prot__{}'.format(regr_gn))
            known_prots.loc[train_omic.index, regr_gn] = regr_obj.\
                    predict_train(cdata_rna, 'prot__{}'.format(regr_gn))

            if i % 27 == 0:
                print('Validation score for {}: {:.2f}'.format(
                    regr_gn,
                    pearsonr(known_prots.loc[train_omic.index, regr_gn],
                             train_prot.flatten())[0]
                    ))

    cdata = TransferDreamCohort(syn, argv[0], intx_types=[argv[1]],
                                cv_seed=171, cv_prop=0.8)

    out_dir = os.path.join(base_dir, 'output_ens')
    out_file = os.path.join(out_dir, argv[0], argv[1], 'results', 'coefs.p')

    # initializes the model and fits it using all of the genes in the
    # `inter`section of the RNA genes, CNA genes, and proteome genes
    clf = mpt.StanEnsemble(intx_type=argv[1], known_prots=known_prots)

    # finds the best combination of model hyper-parameters, uses these
    # parameters to fit to the data
    #clf.tune_coh(cdata, pheno='inter',
    #             tune_splits=4, test_count=4, parallel_jobs=16)
    clf.fit_coh(cdata, pheno='inter') 
    print(clf.eval_coh(cdata, pheno='inter'))

    pickle.dump(clf.get_coef(), open(out_file, 'wb'))


if __name__ == "__main__":
    main(sys.argv[1:])

