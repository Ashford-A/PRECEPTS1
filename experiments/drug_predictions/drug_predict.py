
import sys
sys.path += ['/home/users/grzadkow/compbio/bergamot']

import os
base_dir = os.path.dirname(os.path.realpath(__file__))

from HetMan.features.annot import get_gencode
from HetMan.features.cohorts import *
from HetMan.features.variants import MuType

from HetMan.predict.cross_validation import *
from HetMan.predict.regressors import *

import numpy as np
import pandas as pd

from math import log10
from scipy.stats import ttest_ind
from sklearn.metrics import roc_auc_score

import synapseclient
import cProfile, pstats, io
import pickle


patient_basedir = "/home/exacloud/lustre1/"
patient_files = {
    'SMRT_02299': (patient_basedir + "SMMARTData/Patients/DNA-16-02299/"
                 "results/rsem/rsemOut.genes.results"),
    'SMRT_16113': (patient_basedir + "SMMARTData/Patients/SMMART16113-101/"
                 "results/rsem/rsemOut.genes.results"),
    'PTTB_4409': (patient_basedir + "PTTB/Patients/OPTR4409/OPTR4409T_RNA/"
                  "results/rsem/rsemOut.genes.results"),
    'PTTB_4315': (patient_basedir + "PTTB/Patients/OPTR4315/4315-T-ORG-RNA/"
                  "results/rsem/rsemOut.genes.results"),
    }

patient_cohs = {'SMRT_16113': "TCGA-PRAD",
                'SMRT_02299': "TCGA-PRAD",
                'PTTB_4409': "TCGA-PAAD",
                'PTTB_4315': "TCGA-PAAD"}
tcga_backcohs = {'TCGA-BRCA', 'TCGA-OV', 'TCGA-GBM', 'TCGA-SKCM'}


def main(argv):
    """Runs the experiment."""
    syn = synapseclient.Synapse()
    syn.login()

    # load drug-mutation association data,
    # filter for pan-cancer associations
    drug_mut_assoc = pd.read_csv(
        base_dir + '/../../data/drugs/ioria/drug_anova.txt.gz',
        sep='\t', comment='#'
        )
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
    print(len(pnt_genes))

    # create a VariantCohort with expression only for genes which have
    # point mutations in the drug_mut_assoc dataframe
    # (cv_prop = cross validation proportion)(train on all here)
    # cross val seed is provided as last arg in an HTCondor submit script, and
    # cohort name is the first (should match cohort names as they appear in BMEG)
    tcga_var_coh = VariantCohort(
        syn, cohort=patient_cohs[argv[0]],
        mut_genes=pnt_genes, mut_levels=['Gene', 'Type'],
        cv_seed=int(argv[-1])+1, cv_prop=1
        )

    tcga_back_cohs = {
        coh: VariantCohort(syn, cohort=coh, mut_genes=pnt_genes,
                           mut_levels=['Gene', 'Type'],
                           cv_seed=int(argv[-1])+1, cv_prop=1)
        for coh in tcga_backcohs
        }

    # TODO: recall why frameshifts aren't considered below
    # get list of point mutation types and drugs associated with at least one
    pnt_mtypes = [MuType({('Gene', gn): {
                            ('Type', ('Frame', 'Point')): None}})
                  for gn in pnt_genes]
    pnt_muts = {(gn + '_mut'): mtype for gn, mtype
                in zip(pnt_genes, pnt_mtypes)
                # TODO: the get_samples argument should be a MuTree...right?
                if len(mtype.get_samples(tcga_var_coh.train_mut)) >= 3}
    pnt_drugs = list(set(
        drug_mut_assoc['DRUG'][pnt_indx][drug_mut_assoc['FEAT'][pnt_indx].
                                    isin(pnt_muts.keys())]))
    pnt_drugs.sort()
    print(len(pnt_drugs))

    # ... stores predicted drug responses for cell lines and tcga samples
    ccle_response = {}
    tcga_response = {}
    back_tcga_resp = {coh: {} for coh in tcga_backcohs}

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
    patient_expr = pd.read_csv(patient_files[argv[0]], header=0, sep='\t')

    # get rid of the unnecessary info in gene_id, get Hugo symbols
    patient_expr['gene_id'] = [i.split('^')[1] for i in patient_expr['gene_id']]
    annot_data = get_gencode()
    patient_expr['Symbol'] = [annot_data[gn]['gene_name'] if gn in annot_data
                              else 'no_gene'
                              for gn in patient_expr['gene_id']]

    # ensure that there are no zeros in preparation for log normalization
    patient_expr.loc[:, 'FPKM'] = (
        patient_expr.loc[:, 'FPKM']
        + min(patient_expr.loc[:, 'FPKM'][patient_expr.loc[:,'FPKM'] > 0]) / 2
        )
    # log normalize the FPKM values
    patient_expr.loc[:, 'FPKM'] = np.log2(patient_expr.loc[:,'FPKM'])

    # combine multiple entries of same gene symbol (use their mean)
    patient_expr = patient_expr.groupby(['Symbol'])['FPKM'].mean()
    patient_expr = pd.DataFrame(patient_expr)

    for drug in pnt_drugs:
        drug_clf = eval(argv[1])()
        cell_line_drug_coh = DrugCohort(cohort='ioria', drug_names=[drug],
                                        cv_seed=int(argv[-1]))
        drug_lbl = cell_line_drug_coh.train_resp.columns[0]
        print("Testing drug {} with alias {} ...".format(drug, drug_lbl))

        # TODO: 'Symbol' --> gene_id
        # get the union of genes in all 3 datasets (tcga, ccle, patient/PDM RNAseq
        use_genes = (
            set(tcga_var_coh.genes) & set(cell_line_drug_coh.genes)
            & set(patient_expr.index)
            & reduce(lambda x, y: x & y,
                     [coh.genes for coh in tcga_back_cohs.values()])
            )

        # filter patient (or PDM) RNAseq data to include only use_genes
        patient_expr_filt = patient_expr.loc[use_genes, :]

        # TODO: does patient_expr_filtered need to be transposed?

        # tunes and fits the classifier on the CCLE data, and evaluates its
        # performance on the held-out samples
        pr = cProfile.Profile()
        pr.enable()
        drug_clf.tune_coh(cell_line_drug_coh, pheno=drug_lbl,
                          tune_splits=4, test_count=16,
                          include_genes=use_genes)
        drug_clf.fit_coh(cell_line_drug_coh, pheno=drug_lbl,
                         include_genes=use_genes)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        print(drug_clf)
        clf_perf[drug] = drug_clf.eval_coh(cell_line_drug_coh, pheno=drug_lbl,
                                           include_genes=use_genes)

        # predicts drug response for the patient or PDM, stores classifier
        # for later use
        ccle_response[drug] = pd.Series(
            drug_clf.predict_train(cell_line_drug_coh,
                                   include_genes=use_genes)
            )
        tcga_response[drug] = pd.Series(
            drug_clf.predict_train(tcga_var_coh,
                                   include_genes=use_genes)
            )

        for coh in tcga_backcohs:
            back_tcga_resp[coh][drug] = pd.Series(
                drug_clf.predict_train(tcga_back_cohs[coh],
                                       include_genes=use_genes)
                )

        patient_response[drug] = drug_clf.predict(patient_expr_filt.transpose())[0]

        for gn, mtype in pnt_muts.items():
            print("Gene: {}, Drug: {}".format(gn, drug))
            # for each mutated gene, get the vector of mutation status
            # for the TCGA samples
            mut_stat = np.array(tcga_var_coh.train_pheno(mtype=mtype))

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
                'TCGA_Response': tcga_response,
                'back_TCGA_Response': back_tcga_resp,
                'Patient_Response': patient_response,
                'TCGA_ttest': tcga_ttest, 'TCGA_AUC': tcga_auc}
    out_file = ('/home/users/grzadkow/compbio/bergamot/HetMan/experiments/'
                'drug_predictions/output/mat_' + argv[0] + '_' + argv[1]
                + '__run' + argv[-1] + '.p')
    pickle.dump(out_data, open(out_file, 'wb'))


if __name__ == "__main__":
        main(sys.argv[1:])

