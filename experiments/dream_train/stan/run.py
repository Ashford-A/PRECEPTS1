
# use your own Synapse cache and credentials here
import synapseclient
syn = synapseclient.Synapse()
syn.login()

# loads the challenge data
from HetMan.features.cohorts import TransferDreamCohort
cdata = TransferDreamCohort(
    syn, 'BRCA', intx_types=['controls-expression-of'],
    cv_seed=34, cv_prop=0.8
    )

# initializes the model and fits it using all of the genes in the
# `inter`section of the RNA genes, CNA genes, and proteome genes
import HetMan.predict.bayesian_pathway.multi_protein as mpt
clf = mpt.StanProteinPipe('controls-expression-of')

# finds the best combination of model hyper-parameters, uses these
# parameters to fit to the data
clf.tune_coh(cdata, pheno='inter',
             tune_splits=3, test_count=4, parallel_jobs=12)
clf.fit_coh(cdata, pheno='inter') 

