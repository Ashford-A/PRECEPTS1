
"""
Continuation of drug prediction pipeline. Generates histograms of classifier
performance. Will allow user to specify performance cut_off for each classifier
(this feature is currently hardcoded). Generates boxplots of these well-behaved
classifiers.

Classifier output filenames should follow this template:
    output/SMMART/01234_ElasticNet__run55.p
    output/SMMART/01234_rForest__run55.p
    output/SMMART/01234_SVRrbf__run55.p

Example bash command:
    python quality_assessment.py -t SMMART -s 01234 -r run55

"""

import os
base_dir = os.path.dirname(os.path.realpath(__file__))

import sys
sys.path += [base_dir + '/../../../../bergamot']

from pathlib import Path

import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt
import itertools
import argparse
from HetMan.features.drugs import *


def generate_performance_hists(output_of_1_run, clf_type, plots_dir, id, run):
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
    # i.e. saves to ...plots/01234/01234_ElasticNet_run55_performance_hist.png
    # doesn't take run number into account
    plt.savefig(plots_dir + clf_type + '/' + run + '/'
                + id + '_' + clf_type + '_' + run + '_performance_hist.png')
    plt.close()


# Not yet functional
# TODO: make it rely on a command line arg
# (i.e. choose=False defaults to 0.20, choose=True lets you pick
def choose_cutoff(clf_type, trial, id, run):
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
    print('Performance histogram is saved at plots/' + trial + '/' + id + '/'
          + id + '_' + clf_type + '_' + run + '_performance_hist.png')
    min_clf_perf = float(input("Please specify the minimum classifier performance \n"
                         "allowed (i.e. 0.20) >>>"))
    return min_clf_perf


# sanity checks
# not used
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

# not used
def get_min_and_max_aucs(auc_df):
    minval = auc_df.min().min()
    maxval = auc_df.max().max()

# not yet functional
def calc_pearson_correlation(auc_df, anova_df):
    pass


def generate_drug_mut_assoc_bp(auc_df, anova_df, clf_type, plots_dir, id, run):
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
    plt.savefig(plots_dir + clf_type + '/' + run + '/'
                + id + '_' + clf_type + '_' + run + '_behavior_boxplots.png')
    plt.close(fig)


def generate_ccle_resp_bp(pred_ccle_resp, patient_resp, clf_type, plots_dir, id, run):
    """
    Generates boxplots of (1) predicted and (2) actual cell line responses
    to each drug. Draws a line representative of the predicted patient/sample
    response for comparison.

    pred_ccle_resp (dict):
        key (str): drug name (only those for well behaved classifiers)
        value (pd.Series): vector of cell line predicted responses to drug-key

    patient_resp (pd.Series):
        key (str): drug name
        value (float): predicted patient response to drug-key

    clf_type (str):
        i.e. 'ElasticNet', 'rForest', or 'SVRrbf'

    prefix (str):
        user-specified prefix for output plot file names

    """

    # load CCLE response data relevant to the well-behaving drug classifiers
    actual_ccle_resp = get_drug_ioria(pred_ccle_resp.keys())

    # for each drug make a separate boxplot
    for drug in pred_ccle_resp.keys():
        # TODO: make a line for patient response

        # make a little pandas dataframe
        # TODO: refactor this. i rushed and i'm sure there's a better way.
        single_drug_df = pd.DataFrame({'Measured': actual_ccle_resp[drug].values})
        single_drug_df['Predicted'] = pd.Series(pred_ccle_resp[drug].values)

        fig = plt.figure(figsize=(8, 8))
        bp = single_drug_df.boxplot(showfliers=False, vert=False)
        ax = fig.add_subplot(111)
        ax.grid(False)
        axes = plt.gca()
        plt.title(drug + " responses: actual and predicted by the " + clf_type + " classifier")
        plt.xlabel("AUC")

        # add the points with jitter
        colcount = 0
        for colname in single_drug_df:
            colcount += 1
            y = np.random.normal(colcount, 0.08, single_drug_df.shape[0])
            plt.plot(single_drug_df[colname], y, "c.", alpha=0.2)

        plt.axvline(x=0.50, c="0.75")
        axes.set_xlim([0.0, 1.1])
        axes.set_xticks(np.arange(0, 1.1, 0.1))
        plt.tight_layout()

        plt.savefig(plots_dir + clf_type + '/' + run + '/'
            + id + '_' + clf_type + '_' + drug.replace('/', '__').replace(' ', '_')
            + '_' + run + '_ccle_response_bp.png'
            )
        plt.close(fig)


def pre_main(clf_data, iorio_assoc, plots_dir, id, run):
    clf_method = clf_data['clf_method']

    generate_performance_hists(clf_data, clf_method, plots_dir, id, run)

    # decide R^2 cutoff
    min_clf_perf = 0.20  # or choose_cutoff(clf_type)

    # get list of drugs whose classifiers behave well
    hi_qual_perf = clf_data['Performance'][clf_data['Performance'] > min_clf_perf]

    # remove poorly behaved classifiers from tcga auc data
    hi_qual_tcga_auc = clf_data['TCGA_AUC'].loc[hi_qual_perf.index]

    # remove poorly behaved classifiers from ccle predicted response data
    hi_qual_ccle_resp = {drug: clf_data['CCLE_Response'][drug] for drug in hi_qual_perf.index}

    # remove poorly behaved classifiers from patient response data
    hi_qual_patient_resp = clf_data['Patient_Response'].loc[hi_qual_perf.index]

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
    generate_drug_mut_assoc_bp(our_assoc, iorio_assoc, clf_method, plots_dir, id, run)

    # TODO: generate boxplots of low performance classifiers

    # TODO: get the actual patient response
    # TODO: generate boxplots of ccle predicted and actual response
    generate_ccle_resp_bp(hi_qual_ccle_resp, hi_qual_patient_resp, clf_method, plots_dir, id, run)
    # just in case/couldn't hurt:
    plt.close("all")


def main():

    # take user-specified names of files (located in base_dir + output/)
    parser = argparse.ArgumentParser()

    parser.add_argument('-t', '--trial_dir',
                        # nargs='1',
                        type=str,
                        help='trial directory where output is stored (i.e. SMMART)',
                        )
    parser.add_argument('-s', '--sample_id',
                        # nargs='+',
                        type=str,
                        help='sample identifier (in filename) (i.e. 01234)'
                        )
    parser.add_argument('-r', '--run',
                        # nargs='1',
                        type=str,
                        help='classifier run number (in filename) (i.e. run55)'
                        )

    # TODO: if none provided, throw error
    args = parser.parse_args()

    id = args.sample_id
    trial = args.trial_dir
    run = args.run

    # take output/SMMART/01234_ElasticNet__run55_output.p
    # prefix = args.prefix
    # clf_output_files = args.files
    output_dir = base_dir + '/output/' + trial + '/'
    plots_dir = base_dir + '/plots/' + trial + '/' + id + '/'
    clf_output_files = [filename for filename in os.listdir(output_dir)
                        if id and run in filename]

    # generate
    newdirs = [plots_dir + 'ElasticNet/' + run + '/',
               plots_dir + 'SVRrbf/' + run + '/',
               plots_dir + 'rForest/' + run + '/'
               ]

    for i in newdirs:
        if not os.path.exists(i):
            path = Path(i)
            path.mkdir(parents=True)

    elast_data = None
    rfor_data = None
    svr_data = None

    for filename in clf_output_files:
        if 'ElasticNet' in filename:
            elast_data = pickle.load(open(output_dir + filename, 'rb'))
            elast_data['clf_method'] = "ElasticNet"
        if 'rForest' in filename:
            rfor_data = pickle.load(open(output_dir + filename, 'rb'))
            rfor_data['clf_method'] = "rForest"
        if 'SVRrbf' in filename:
            svr_data = pickle.load(open(output_dir + filename, 'rb'))
            svr_data['clf_method'] = "SVRrbf"
        # TODO: throw error if none of these 3 are in file name

    # read in Iorio et al's drug-mutation associations (AUC)
    iorio_assoc = pd.read_csv(
        base_dir + "/../../data/drugs/ioria/drug_anova.txt.gz",
        delimiter='\t', comment='#', usecols=['FEAT', 'DRUG', 'PANCAN']
        )

    # keep only the point mutations in iorio association dataset
    iorio_assoc = iorio_assoc[iorio_assoc['FEAT'].str.contains('_mut')]
    iorio_assoc = iorio_assoc.pivot(index='FEAT', columns='DRUG', values='PANCAN')

    if elast_data is not None:
        pre_main(elast_data, iorio_assoc, plots_dir, id, run)
    if rfor_data is not None:
        pre_main(rfor_data, iorio_assoc, plots_dir, id, run)
    if svr_data is not None:
        pre_main(svr_data, iorio_assoc, plots_dir, id, run)

    # again, just in case
    plt.close('all')


if __name__ == "__main__":
    main()
