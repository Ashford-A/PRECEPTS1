
import sys
import pickle
sys.path += ['/home/users/grzadkow/compbio/scripts']

from HetMan.predict.cross_validation import *
from HetMan.features.cohorts import *
from HetMan.predict.pipelines import *
# MG: should be from HetMan.features.variants import MuType
from HetMan.mutation import MuType # ?

import numpy as np
import pandas as pd

from math import log10
from scipy.stats import ttest_ind
from sklearn.metrics import roc_auc_score

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
    tcga_var_coh = VariantCohort(syn, cohort=argv[0], mut_genes=pnt_genes,
                   mut_levels=['Gene', 'Form', 'Protein'],
                   cv_seed=int(argv[-1])+1, cv_prop=1)

    # TODO: recall why frameshifts aren't considered below
    # get list of point mutation types and drugs associated with at least one
    pnt_mtypes = [
        MuType({('Gene', gn):
                {('Form', ('Nonsense_Mutation', 'Missense_Mutation')): None}}
              ) for gn in pnt_genes]
    pnt_muts = {(gn + '_mut'):mtype for gn,mtype
                in zip(pnt_genes, pnt_mtypes)
                # TODO: the get_samples argument should be a MuTree...right?
                if len(mtype.get_samples(tcga_var_coh.train_omics())) >= 3}
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

    # loads patient (or patient-derived model (PDM)) RNAseq data
    patient_expr = pd.read_csv(
        '/home/users/grzadkow/compbio/input-data/rnaseq_4315.csv',
        header=0)

    for drug in pnt_drugs:
        print("Testing drug " + drug + " ....")
        drug_clf = eval(argv[1])()

        # TODO: check on unexpected args
        # loads cell line drug response and array expression data
        cell_line_drug_coh = DrugCohort(drug, source='ioria', random_state=int(argv[-1]))


        # TODO: Do the normalization of the patient RPKMs HERE using all genes available in patient expression data
        patient_expr.loc[:, 'RPKM'] = (patient_expr.loc[:, 'RPKM']
                                       + min(patient_expr.loc[:, 'RPKM'][patient_expr.loc[:,'RPKM'] > 0]) / 2)
        patient_expr = patient_expr.groupby(['Symbol'])['RPKM'].mean()
        patient_expr = pd.DataFrame(patient_expr)

        # get the union of genes in all 3 datasets (tcga, ccle, patient/PDM RNAseq
        use_genes = (set(tcga_var_coh.genes) & set(cell_line_drug_coh.genes) & set(patient_expr.ix['Symbol']))

        # filter patient (or PDM) RNAseq data to include only use_genes
        patient_expr_filtered = patient_expr.ix[patient_expr['Symbol'].isin(use_genes),:]
        # or should i do something like the following? figure out what patient_expr_filtered actually looks like...
        # patient_expr_filtered = patient_expr_filtered.loc[:, use_genes]

        # TODO: verify that transposition needs to happen here (and not earlier)
        patient_expr_filtered = patient_expr_filtered.transpose()

        # tunes and fits the classifier on the CCLE data, and evaluates its
        # performance on the held-out samples
        drug_clf.tune_coh(cell_line_drug_coh, include_genes=use_genes)
        drug_clf.fit_coh(cell_line_drug_coh, include_genes=use_genes)
        clf_perf[drug] = drug_clf.eval_coh(cell_line_drug_coh, include_genes=use_genes)

        # predicts drug response for the patient or PDM, stores classifier
        # for later use
        ccle_response[drug] = pd.Series(drug_clf.predict_train(cell_line_drug_coh, include_genes=use_genes))
        tcga_response[drug] = pd.Series(drug_clf.predict_train(tcga_var_coh, include_genes=use_genes))
        patient_response[drug] = drug_clf.predict(patient_expr_filtered)[0]


        for gn, mtype in pnt_muts.items():
            # for each mutated gene, get the vector of mutation status
            # for the TCGA samples
            mut_stat = np.array(
                # TODO: make sure the following is doing what it's supposed to do...
                # MG: unless I fucked up the Independence Day Refactoring it should work
                tcga_var_coh.train_pheno(mtype=mtype)
                # was:
                # cdata.train_mut_.status(cdata.train_expr_.index,
                #                       mtype=mtype)
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


