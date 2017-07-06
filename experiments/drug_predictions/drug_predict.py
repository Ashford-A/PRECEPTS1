
import sys
import pickle
sys.path += ['/home/users/grzadkow/compbio/scripts']

from HetMan.predict.cross_validation import *
from HetMan.features.cohorts import *
from HetMan.predict.pipelines import *
from HetMan.mutation import MuType # ?

import numpy as np
import pandas as pd

from math import log10
from scipy.stats import ttest_ind
from sklearn.metrics import roc_auc_score
from copy import deepcopy

base_dir = ('/home/users/grzadkow/compbio/scripts/HetMan/'
            'experiments/drug_predictions')


def main(argv):
    """Runs the experiment."""
    syn = synapseclient.Synapse()
    syn.login('grzadkow', 'W0w6g1i8A')

    # load drug-mutation association data,
    # filter for pan-cancer associations
    drug_mut_assoc = pd.read_csv(base_dir + '/input/drug_data.txt',
                            sep='\t', comment='#')
    drug_mut_assoc = drug_mut_assoc.ix[drug_mut_assoc['PANCAN'] != 0, :]

    # categorize associations by mutation type
    pnt_indx = drug_mut_assoc['FEAT'].str.contains('_mut$')
    # TODO: determine how iorio handled CNVs (they're currently ignored)
    cnv_indx = drug_mut_assoc['FEAT'].str.contains('^(?:loss|gain):')
    fus_indx = drug_mut_assoc['FEAT'].str.contains('_fusion$')

    # get list of genes affected by point mutations, load TCGA cohort
    # with corresponding set of mutations
    pnt_genes = list(set(
        x[0] for x in drug_mut_assoc['FEAT'][pnt_indx].str.split('_')))

    # create a VariantCohort with expression only for genes which have
    # point mutations in the drug_mut_assoc dataframe
    # (cv_prop = cross validation proportion)(train on all here)
    # cross val seed is provided as last arg in an HTCondor submit script, and
    # cohort name is the first (should match cohort names as they appear in BMEG)
    # TODO: think about how to handle mut_genes and include_genes
    tcga_var_coh = VariantCohort(syn, cohort=argv[0], mut_genes=pnt_genes,
                   mut_levels=['Gene', 'Form', 'Protein'],
                   cv_seed=int(argv[-1])+1, cv_prop=1)
    tcga_coh_expr = deepcopy(tcga_var_coh.train_omics())

    # TODO: recall why frameshifts aren't considered below
    # get list of point mutation types and drugs associated with at least one
    pnt_mtypes = [
        MuType({('Gene', gn):
                {('Form', ('Nonsense_Mutation', 'Missense_Mutation')): None}}
              ) for gn in pnt_genes]
    pnt_muts = {(gn + '_mut'):mtype for gn,mtype
                in zip(pnt_genes, pnt_mtypes)
                if len(mtype.get_samples(tcga_var_coh.train_mut_)) >= 3}
    pnt_drugs = list(set(
        drug_mut_assoc['DRUG'][pnt_indx][drug_mut_assoc['FEAT'][pnt_indx].
                                    isin(pnt_muts.keys())]))

    # ... stores predicted drug responses for cell lines and tcga samples
    ccle_response = {}
    tcga_response = {}

    # ... stores predicted drug response for organoid sample
    patient_response = pd.Series(float('nan'), index=pnt_drugs)

    # array that stores classifier performance on held-out cell lines
    clf_perf = pd.Series(float('nan'), index=pnt_drugs)

    # ... stores t-test p-values for mutation state vs predicted
    # drug responses in TCGA cohort
    tcga_ttest = pd.DataFrame(float('nan'),
                              index=pnt_drugs, columns=pnt_muts.keys())

    # ... stores AUC scores for mutation vs drug response in TCGA
    tcga_auc = pd.DataFrame(float('nan'),
                            index=pnt_drugs, columns=pnt_muts.keys())

    # loads organoid RNAseq data
    pred_data = pd.read_csv(
        '/home/users/grzadkow/compbio/input-data/rnaseq_4315.csv',
        header=0)

    for drug in pnt_drugs:
        print("Testing drug " + drug + " ....")
        drug_clf = eval(argv[1])()

        # TODO: find the union of genes between drug cohort and variant cohort
        #       instead of the tcga_var_coh stuff.
        # loads cell line drug response and array expression data
        coh = DrugCohort(drug, source='ioria', random_state=int(argv[-1]))
        tcga_var_coh.train_expr_ = deepcopy(tcga_coh_expr)
        use_genes = (set(tcga_var_coh.train_expr_.columns)
                     & set(coh.drug_expr.columns))

        # pred_data = patient expression data (change to patient_expr or something)
        # pred_samp = subset of pred_data with genes that we want
        #   (i.e. just genes we want)
        # processes organoid RNAseq data to match TCGA and drug response data
        pred_samp = pred_data.ix[pred_data['Symbol'].isin(use_genes), :]
        # TODO: add a normalization step
        pred_samp.loc[:, 'RPKM'] = (
            pred_samp.loc[:, 'RPKM']
            + min(pred_samp.loc[:, 'RPKM'][pred_samp.loc[:,'RPKM'] > 0]) / 2)
        pred_samp = pred_samp.groupby(['Symbol'])['RPKM'].mean()
        pred_samp = pd.DataFrame(pred_samp).transpose()

        # just use genes in all: drug_coh, variant_coh, patient_expr/pred_data
        use_genes &= set(pred_samp.columns)
        pred_samp = pred_samp.loc[:, use_genes]

        # TODO: pass include/exclude genes args to tune, fit, etc. as in the first here...
        # tunes and fits the classifier on the CCLE data, and evaluates its
        # performance on the held-out samples
        drug_clf.tune_coh(coh, include_genes=use_genes)
        coh.tune_clf(drug_clf)
        coh.fit_clf(drug_clf)
        clf_perf[drug] = coh.eval_clf(drug_clf)

        # predicts drug response for the organoid, stores classifier
        # for later use
        # TODO: do include_genes=use_genes for similar examples
        # TODO: tcga_var_coh should be tcga_coh, coh should be cellline_coh or something
        ccle_response[drug] = pd.Series(drug_clf.predict_train(coh, include_genes=use_genes))
        tcga_response[drug] = pd.Series(drug_clf.predict_train(tcga_var_coh, include_genes=use_genes))
        patient_response[drug] = drug_clf.predict(pred_samp)[0]

        for gn, mtype in pnt_muts.items():
            # for each mutated gene, get the vector of mutation status
            # for the TCGA samples
            mut_stat = np.array(
                tcga_var_coh.train_mut_.status(tcga_var_coh.train_expr_.index,
                                        mtype=mtype)
                )

            # gets the classifier's predictions of drug response for the
            # TCGA cohort, and evaluate its concordance with mutation status
            tcga_ttest.loc[drug, gn] = -log10(
                ttest_ind(tcga_response[drug][mut_stat],
                          tcga_response[drug][~mut_stat],
                          equal_var=False)[1]
                )
            tcga_auc.loc[drug, gn] = roc_auc_score(mut_stat,
                                                   tcga_response[drug])

    # save everything to file
    out_data = {'Performance': clf_perf, 'CCLE_Response': ccle_response,
                'TCGA_Response': tcga_response, 'Patient_Response': patient_response,
                'TCGA_ttest': tcga_ttest, 'TCGA_AUC': tcga_auc}
    out_file = ('/home/users/grzadkow/compbio/scripts/HetMan/experiments/'
                'drug_predictions/output/mat_' + argv[0] + '_' + argv[1]
                + '__run' + argv[-1] + '.p')
    pickle.dump(out_data, open(out_file, 'wb'))


if __name__ == "__main__":
        main(sys.argv[1:])


