
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
    tcga_var_coh = VariantCohort(syn, cohort=argv[0], mut_genes=pnt_genes,
                   mut_levels=['Gene', 'Form', 'Protein'],
                   cv_seed=int(argv[-1])+1, cv_prop=1)
    # returns the expression dataset without any filtering (genes have already
    # been filtered above with mut_genes=pnt_genes)
    tcga_coh_expr = tcga_var_coh.train_omics()

    # TODO: recall why frameshifts aren't considered below
    # get list of point mutation types and drugs associated with at least one
    pnt_mtypes = [
        MuType({('Gene', gn):
                {('Form', ('Nonsense_Mutation', 'Missense_Mutation')): None}}
              ) for gn in pnt_genes]
    pnt_muts = {(gn + '_mut'):mtype for gn,mtype
                in zip(pnt_genes, pnt_mtypes)
                # TODO: the get_samples argument should be a MuTree...right?
                if len(mtype.get_samples(tcga_coh_expr)) >= 3}
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

        # loads cell line drug response and array expression data
        cell_line_drug_coh = DrugCohort(drug, source='ioria', random_state=int(argv[-1]))
        cell_line_coh_expr = cell_line_drug_coh.train_omics()
        # TODO: verify that ".columns" not necessary at end of each expr obj
        # get the union of genes in the datasets
        use_genes = (set(tcga_coh_expr) & set(cell_line_coh_expr))

        # filter patient (or PDM) RNAseq data to include only genes which are present in both
        # the cell line drug response cohort and TCGA cohort
        patient_expr_filtered = patient_expr.ix[patient_expr['Symbol'].isin(use_genes), :]
        # TODO: add a normalization step
        patient_expr_filtered.loc[:, 'RPKM'] = (
            patient_expr_filtered.loc[:, 'RPKM']
            + min(patient_expr_filtered.loc[:, 'RPKM'][patient_expr_filtered.loc[:,'RPKM'] > 0]) / 2)
        patient_expr_filtered = patient_expr_filtered.groupby(['Symbol'])['RPKM'].mean()
        patient_expr_filtered = pd.DataFrame(patient_expr_filtered).transpose()

        # just use genes in all: drug_coh, variant_coh, patient_expr/patient_expr
        use_genes &= set(patient_expr_filtered.columns)
        patient_expr_filtered = patient_expr_filtered.loc[:, use_genes]

        # TODO: pass include/exclude genes args to tune, fit, etc. as in the first here...
        # tunes and fits the classifier on the CCLE data, and evaluates its
        # performance on the held-out samples
        drug_clf.tune_coh(cell_line_drug_coh, include_genes=use_genes)
        cell_line_drug_coh.tune_clf(drug_clf)
        cell_line_drug_coh.fit_clf(drug_clf)
        clf_perf[drug] = cell_line_drug_coh.eval_clf(drug_clf)

        # predicts drug response for the organoid, stores classifier
        # for later use
        # TODO: do include_genes=use_genes for similar examples
        ccle_response[drug] = pd.Series(drug_clf.predict_train(cell_line_drug_coh, include_genes=use_genes))
        tcga_response[drug] = pd.Series(drug_clf.predict_train(tcga_var_coh, include_genes=use_genes))
        patient_response[drug] = drug_clf.predict(patient_expr_filtered)[0]

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


