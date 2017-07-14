
"""
Continuation of drug prediction pipeline. Generates histograms of classifier
performance. Will allow user to specify performance cut_off for each classifier
(this feature is currently hardcoded). Generates boxplots of these well-behaved
classifiers.

example bash command:
    python HetMan/experiments/drug_predictions/quality_assessment.py -f \
    mat_SMRT_1234_ElasticNet__run55.p \
    mat_SMRT_1234_rForest__run55.p \
    mat_SMRT_1234_SVRrbf__run55.p \
    -p SMRT_1234
"""

import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import itertools
import argparse

basedir = '/Users/manningh/PycharmProjects/bergamot/' \
              'HetMan/experiments/drug_predictions/'


def generate_performance_hists(output_of_1_run, clf_type):
    # (performance is given in R^2

    perf = output_of_1_run['Performance']

    # allow for user-specified bin size?
    binsize = 0.05

    plt.figure()
    performance_hist = plt.hist(perf,
                                bins=np.arange(min(perf),
                                               max(perf) + binsize, binsize))
    plt.xlim(0.0,max(perf) + binsize)
    plt.ylabel('Number of Classifiers')
    plt.xlabel('Classifier Performance')
    plt.title(clf_type + 'with bin-size of ' + str(binsize))
    plt.savefig(basedir + 'plots/' + prefix + clf_type + 'performance_hist.png')
    plt.close()

def choose_cutoff(clf_type):
    """
    Describes location of the performance histogram for a given classifier run.
    Returns user-specified minimum classifier performance (in R^2).
    Only drug response predictions from acceptably performing classifiers
    will be used.

    Params:
        clf_type (str): type of classifier ('elast', 'rfor', or 'svr')
    Returns:
        min_clf_perf (float): minimum classifier performance (R^2)
    """
    print('Performance histogram is saved at ' + clf_type + '/' + prefix + '/performance_hist.png')
    min_clf_perf = float(input("Please specify the minimum classifier performance \n"
                         "allowed (i.e. 0.20) >>>"))
    return min_clf_perf

# sanity checks
def generate_mean_AUC_hists(auc_df):
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

def generate_drug_mut_assoc_boxplots(auc_df, anova_df, clf_type):
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
    assoc_directions = {drug: {'pos_anova': [], 'neg_anova': []} for drug in anova_df.columns}

    for drug in anova_df.columns:
        row_indexer = 0
        for anova_score in anova_df[drug]:
            if anova_score < 0:
                mut = anova_df.index[row_indexer]
                assoc_directions[drug]['neg_anova'].append(mut)
            if anova_score > 0:
                mut = anova_df.index[row_indexer]
                assoc_directions[drug]['pos_anova'].append(mut)
            row_indexer += 1

    # generate boxplots of tcga_auc
    fig = plt.figure(figsize=(14, 8))
    bp = auc_df.boxplot(showfliers=False)
    ax = fig.add_subplot(111)
    ax.grid(False)
    axes = plt.gca()
    plt.title(clf_type + "classifier behavior")
    plt.ylabel("AUC")
    plt.xlabel("Drug Classifier")
    colcount = 0
    for drug in auc_df.columns:
        colcount += 1
        # lists muts
        pos_anova_muts = assoc_directions[drug]['pos_anova']  # color these blue
        neg_anova_muts = assoc_directions[drug]['neg_anova']  # color these red
        zero_anova_muts = auc_df[drug].index.drop(pos_anova_muts).drop(neg_anova_muts)
        # get the drug column (type is pd.Series)
        aucs = auc_df[drug]
        # generate series for different data colors (auc values)
        blue_points = aucs[pos_anova_muts]
        red_points = aucs[neg_anova_muts]
        black_points = aucs[zero_anova_muts]
        colors = itertools.cycle(["r.", "k.", "b."])
        point_size = itertools.cycle([9.0, 3.0, 9.0])
        for colorgroup in [red_points, black_points, blue_points]:
            # prepare to add jitter
            x = np.random.normal(colcount, 0.08, len(colorgroup))
            plt.plot(x, colorgroup, next(colors), alpha=0.6, markersize=next(point_size))
        plt.xticks(rotation='45')
    plt.axhline(y=0.50, c="0.75")
    axes.set_ylim([0.0, 1.0])
    axes.set_yticks(np.arange(0, 1.1, 0.1))
    plt.tight_layout()
    # plt.show(block=False)
    # plt.waitforbuttonpress(0)
    # print("Press any button to close the plots and proceed.")
    plt.savefig(basedir + 'plots/' + prefix + clf_type + 'behavior_boxplots.png')
    plt.close(fig)

def generate_response_boxplot(tcga_response, patient_response):
    pass


def pre_main(clf_data, iorio_assoc):
    clf_method = clf_data['clf_method']

    generate_performance_hists(clf_data, clf_method)

    # decide R^2 cutoff
    min_clf_perf = 0.20  # choose_cutoff(clf_type)

    # remove poorly behaving drug classifiers from data
    hi_qual_perf = clf_data['Performance'][clf_data['Performance'] > min_clf_perf]
    hi_qual_tcga_auc = clf_data['TCGA_AUC'].loc[hi_qual_perf.index]

    # consider sanity check: generate_mean_AUC_hists(hi_qual_tcga_auc)

    # get them into the right format (our_assoc.columns = drugs, our_assoc.index = muts)
    our_assoc = hi_qual_tcga_auc.transpose()
    our_assoc.index.name = 'FEAT'
    our_assoc.columns.name = 'DRUG'

    # get union of point mutations, union of drugs
    shared_muts = list(set(our_assoc.index) & set(iorio_assoc.index))
    shared_drugs = list(set(our_assoc.columns) & set(iorio_assoc.columns))

    # pare down the association dataframes with respect to shared drugs, muts
    iorio_assoc = iorio_assoc[shared_drugs].loc[shared_muts]
    our_assoc = our_assoc[shared_drugs].loc[shared_muts]

    # generate boxplots of high performance classifiers
    generate_drug_mut_assoc_boxplots(our_assoc, iorio_assoc, clf_method)

    # TODO: generate boxplots of low performance classifiers

    # just in case/couldn't hurt:
    plt.close("all")

def main():

    # take user-specified names of files (located in basedir + output/)
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--files', nargs='+')
    # prefix could be patient number
    parser.add_argument('-p', '--prefix')
    args = parser.parse_args()

    prefix = args.prefix
    clf_output_files = args.files

    # TODO: if none provided, throw error
    elast_data = None
    rfor_data = None
    svr_data = None

    for filename in clf_output_files:
        if 'ElasticNet' in filename:
            elast_data = pickle.load(open(basedir + 'output/' + filename, 'rb'))
            elast_data['clf_method'] = "elast"
        if 'rForest' in filename:
            rfor_data = pickle.load(open(basedir + 'output/' + filename, 'rb'))
            rfor_data['clf_method'] = "rfor"
        if 'SVRrbf' in filename:
            svr_data = pickle.load(open(basedir + 'output/' + filename, 'rb'))
            svr_data['clf_method'] = "svr"

    # read in Iorio et al's drug-mutation associations (AUC)
    iorio_assoc = pd.read_csv(basedir + "input/drug_data.txt",
                                       delimiter='\t',
                                       comment='#',
                                       usecols=['FEAT', 'DRUG', 'PANCAN']
                                       )
    # keep only the point mutations in iorio association dataset
    iorio_assoc = iorio_assoc[iorio_assoc['FEAT'].str.contains('_mut')]
    iorio_assoc = iorio_assoc.pivot(index='FEAT', columns='DRUG', values='PANCAN')

    if elast_data is not None:
        pre_main(elast_data, iorio_assoc)
    if rfor_data is not None:
        pre_main(rfor_data, iorio_assoc)
    if svr_data is not None:
        pre_main(svr_data, iorio_assoc)

if __name__ == "__main__":
    main()