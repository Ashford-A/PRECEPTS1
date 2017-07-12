
"""
Continuation of drug prediction pipeline. Assesses quality of classifier
performance.

HM's notes to self (forgive the dumb wording)

1. Plots performance of drug classifiers
2. Proceeds with high quality classifiers (based on determined by R^2)
3. Converts matrix of drug-mutation associations (i.e. mat_TCGA-BRCA_ElasticNet__run55.p)
    to binary
        is the presence/absence of this mutation strongly correlated
        with each drug classifier? (yes or no)
4. Compares that new binary matrix to Ioria's binary matrix of muts and drugs
    (drug_data.txt)
    NOTE: 3 and 4 are only performed for point mutations in pan-cancer data
        -calculate false negs and false positives/(precision v. recall)
5. compare patient's predicted drug responses to those predicted for TCGA
    (i.e. where does our patient fall with respect to the other TCGA patients with
    this or that diagnosis?)

"""

import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def choose_cutoff(output_of_1_run):
    # (performance is given in R^2
    # abs_perf = list(map(abs,elast_data_r55['Performance']))

    perf = elast_data_r55['Performance']
    # allow for user-specified bin size?
    binsize = 0.05

    plt.figure()
    performance_hist = plt.hist(perf,
                                bins=np.arange(min(perf),
                                               max(perf) + binsize, binsize))
    plt.xlim(0.0,max(perf) + binsize)
    plt.ylabel('Number of Classifiers')
    plt.xlabel('Classifier Performance')
    plt.title('Bin-size of ' + str(binsize))
    plt.show(block=False)

    plt.waitforbuttonpress(0)
    print("Press any button to close this plot and proceed.")
    plt.close()

    min_clf_perf = float(input("Please specify the minimum classifier performance \n"
                         "allowed in future calculations (i.e. 0.20) "))

    return min_clf_perf

def show_mean_AUC_hists(auc_df):
    # get/plot mean auc for each mutation across all drug clfs
    plt.figure()
    plt.hist(auc_df.mean(0))
    plt.title("Mean AUC for each mut across all drug clfs")
    plt.ylabel("Count of mutations with the respective AUC")
    plt.xlabel("Mean AUC")
    plt.show(block=False)

    # do so for each drug across all muts
    plt.figure()
    plt.hist(auc_df.mean(1))
    plt.title("Mean AUC for each drug clf across all mutations")
    plt.ylabel("Count of clfs with the respective AUC")
    plt.xlabel("Mean AUC")
    plt.show(block=False)

    plt.waitforbuttonpress(0)
    print("Press any button to close the plots and proceed.")
    plt.close("all")


def get_min_and_max_aucs(auc_df):
    minval = auc_df.min().min()
    maxval = auc_df.max().max()


def calc_pearson_correlation(auc_df, anova_df):
    pass

def show_drug_mut_assoc_boxplots(auc_df, anova_df):
    """
    Generates a figure with 1 boxplot per high-quality classifier.
    Data points for each boxplot are drug-mutation associations from auc_df.
    They are colored according to whether their are expected to be
    positively correlated (blue), negatively correlated (green),
    or not correlated (black) as per the ANOVA scores in anova_df.

    auc_df and anova_df should bear the same shape:
        cols = mutations
        rows = drugs
    """

    # get the sign ("direction") of the ANOVA score in iorio_assoc
    # this will be used to determine the color of datapoints in boxplots
    assoc_directions = {drug: {'pos_anova': [], 'neg_anova': []} for drug in iorio_assoc.columns}

    for drug in iorio_assoc.columns:
        row_indexer = 0
        for anova_score in iorio_assoc[drug]:
            if anova_score < 0:
                mut = iorio_assoc.index[row_indexer]
                assoc_directions[drug]['neg_anova'].append(mut)
            if anova_score > 0:
                mut = iorio_assoc.index[row_indexer]
                assoc_directions[drug]['pos_anova'].append(mut)
            row_indexer += 1

    # generate boxplots of tcga_auc
    fig = plt.figure(figsize=(14, 8))
    bp = our_assoc.boxplot(showfliers=False)
    ax = fig.add_subplot(111)
    ax.grid(False)
    axes = plt.gca()
    plt.title("Very informative title")
    plt.ylabel("AUC")
    plt.xlabel("Drug Classifier")
    colcount = 0
    for drug in our_assoc.columns:
        colcount += 1
        pos_anova_muts = assoc_directions[drug]['pos_anova']  # color these blue
        neg_anova_muts = assoc_directions[drug]['neg_anova']  # color these red
        # get the drug column (type is pd.Series)
        aucs = our_assoc[drug]
        # prepare to add jitter
        x = np.random.normal(colcount, 0.08, len(aucs))
        plt.plot(x, aucs, 'b.', alpha=0.2)
        plt.xticks(rotation='45')
    plt.axhline(y=0.50, c="r")
    axes.set_ylim([0.0, 1.0])
    axes.set_yticks(np.arange(0, 1.1, 0.1))
    plt.tight_layout()
    plt.show(block=False)

    plt.waitforbuttonpress(0)
    print("Press any button to close the plots and proceed.")
    plt.close("all")

def main():

    basedir = '/Users/manningh/PycharmProjects/bergamot/' \
              'HetMan/experiments/drug_predictions/'

    # wasn't there a util function for loading output of many runs? a donde fue?
    elast_data_r55 = pickle.load(open(basedir +
                                      'output/mat_TCGA-BRCA_ElasticNet__run55.p',
                                      'rb'
                                      )
                                 )
    # read in Iorio et al's drug-mutation associations (AUC)
    iorio_assoc = pd.read_csv(basedir + "input/drug_data.txt",
                                       delimiter='\t',
                                       comment='#',
                                       usecols=['FEAT', 'DRUG', 'PANCAN']
                                       )

    # TODO: why is this making it say the dataframe is empty...despite being filled + having shape?
    # seems to pivot though
    # keep only the point mutations in iorio association dataset
    iorio_assoc = iorio_assoc[iorio_assoc['FEAT'].str.contains('_mut')]

    # determine a performance cut-off (in terms of acceptable R^2 values)
    min_clf_perf = choose_cutoff(elast_data_r55)

    # remove poorly performing drug classifiers from data
    hi_qual_perf = elast_data_r55['Performance'][elast_data_r55['Performance'] > min_clf_perf]
    hi_qual_tcga_auc = elast_data_r55['TCGA_AUC'].loc[hi_qual_perf.index]

    # show_mean_AUC_hists(hi_qual_tcga_auc)

    # get them into the right format (our_assoc.columns = drugs, our_assoc.index = muts)
    our_assoc = hi_qual_tcga_auc.transpose()
    our_assoc.index.name = 'FEAT'
    our_assoc.columns.name = 'DRUG'

    iorio_assoc = iorio_assoc.pivot(index='FEAT', columns='DRUG', values='PANCAN')


    # get union of point mutations, union of drugs
    shared_muts = list(set(our_assoc.index) & set(iorio_assoc.index))
    shared_drugs = list(set(our_assoc.columns) & set(iorio_assoc.columns))

    # pare down the association dataframes with respect to shared drugs, muts
    iorio_assoc = iorio_assoc[shared_drugs].loc[shared_muts]
    our_assoc = our_assoc[shared_drugs].loc[shared_muts]

    '''
    # sanity check: calculate the mean number of non NaN values per drug in iorio_assoc
    # i don't think it will be enough to correlate anything with...
    mylist = []
    for col in iorio_assoc:
        mylist.append(iorio_assoc[col].count())
    # np.mean(mylist) = 1.95
    '''


if __name__ == "__main__":
    main()