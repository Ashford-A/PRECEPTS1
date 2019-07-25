
import os
import sys

base_dir = os.path.join(os.environ['DATADIR'], 'HetMan', 'dyad_infer')
sys.path.extend([os.path.join(os.path.dirname(__file__), '../../..')])
plot_dir = os.path.join(base_dir, 'plots', 'experiment')

from HetMan.experiments.dyad_infer import *
import argparse
import bz2
import dill as pickle
import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


def plot_label_stability(infer_df, stat_dict, auc_dict, args):
    fig, axarr = plt.subplots(figsize=(13, 8), nrows=2, ncols=4)

    auc_df = pd.DataFrame.from_dict(auc_dict, orient='index')
    auc_bins = pd.qcut(auc_df.values.flatten(), 4, precision=7).categories
    stat_clrs = {'00': '0.31', '10': 'red', '01': 'blue', '11': 'green'}

    lbl_dfs = {
        norm_lbl: {auc_bin: {stat_str: pd.DataFrame(columns=['Mean', 'Var'])
                             for stat_str in ['00', '10', '01', '11']}
                   for auc_bin in auc_bins}
        for norm_lbl in ['Raw', 'Norm']
        }

    for (mtype1, mtype2), (val_arr1, val_arr2) in infer_df.iterrows():
        stat_vec = np.array([''.join([str(int(y)) for y in x])
                             for x in zip(stat_dict[mtype1],
                                          stat_dict[mtype2])])

        wt_mean1 = np.mean(np.concatenate(
            np.array(val_arr1)[stat_vec == '00']))
        wt_mean2 = np.mean(np.concatenate(
            np.array(val_arr2)[stat_vec == '00']))

        mut_mean1 = np.mean(np.concatenate(
            np.array(val_arr1)[stat_vec == '10']))
        mut_mean2 = np.mean(np.concatenate(
            np.array(val_arr2)[stat_vec == '01']))

        for stat_str, vals1 in zip(stat_vec, val_arr1):
            use_bin = auc_bins[auc_bins.get_loc(
                auc_dict[mtype1, mtype2]['Mtype1'])]
            norm_vals1 = [(val1 - wt_mean1) / (mut_mean1 - wt_mean1)
                          for val1 in vals1]

            lbl_dfs['Raw'][use_bin][stat_str] = lbl_dfs['Raw'][use_bin][
                stat_str].append({'Mean': np.mean(vals1),
                                  'Var': np.var(vals1, ddof=1)},
                                 ignore_index=True)

            lbl_dfs['Norm'][use_bin][stat_str] = lbl_dfs['Norm'][use_bin][
                stat_str].append({'Mean': np.mean(norm_vals1),
                                  'Var': np.var(norm_vals1, ddof=1)},
                                 ignore_index=True)

        for stat_str, vals2 in zip(stat_vec, val_arr2):
            use_bin = auc_bins[auc_bins.get_loc(
                auc_dict[mtype1, mtype2]['Mtype2'])]
            norm_vals2 = [(val2 - wt_mean2) / (mut_mean2 - wt_mean2)
                          for val2 in vals2]

            lbl_dfs['Raw'][use_bin][stat_str] = lbl_dfs['Raw'][use_bin][
                stat_str][::-1].append({'Mean': np.mean(vals2),
                                        'Var': np.var(vals2, ddof=1)},
                                       ignore_index=True)

            lbl_dfs['Norm'][use_bin][stat_str] = lbl_dfs['Norm'][use_bin][
                stat_str][::-1].append({'Mean': np.mean(norm_vals2),
                                        'Var': np.var(norm_vals2, ddof=1)},
                                       ignore_index=True)

    for i, norm_lbl in enumerate(['Raw', 'Norm']):
        for j, auc_bin in enumerate(auc_bins):
            if i == 0:
                axarr[i, j].set_title(auc_bin)

            for stat_str in ['00', '10', '01', '11']:
                use_df = lbl_dfs[norm_lbl][auc_bin][stat_str]

                if stat_str == '11':
                    mean_wind = 11
                else:
                    mean_wind = 4

                if use_df.shape[0] > 0:
                    qnt_vec = use_df.Mean.quantile(
                        q=np.linspace(0.025, 0.975, 380)).values

                    qnt_df = pd.DataFrame([
                        use_df.Var[(use_df.Mean >= qnt_vec[i - mean_wind])
                                   & (use_df.Mean
                                      < qnt_vec[i + mean_wind])].quantile(
                                          q=[0.25, 0.5, 0.75])
                        for i in range(mean_wind, len(qnt_vec) - mean_wind)
                        ])

                    if stat_str in ['00', '10']:
                        use_ls = '-'
                        use_lw = 3.1

                    else:
                        use_ls = '--'
                        use_lw = 1.9

                    if norm_lbl == 'Raw':
                        use_yfnc = np.log10
                    else:
                        use_yfnc = lambda x: x

                    axarr[i, j].plot(qnt_vec[mean_wind:-mean_wind],
                                     use_yfnc(qnt_df[0.5]),
                                     color=stat_clrs[stat_str],
                                     linewidth=use_lw, alpha=0.47,
                                     linestyle=use_ls)

            if norm_lbl == 'Norm':
                axarr[i, j].set_ylim([0, axarr[i, j].get_ylim()[1] * 1.07])

    fig.tight_layout(w_pad=1.7, h_pad=2.3)
    plt.savefig(
        os.path.join(plot_dir,
                     "{}__samps-{}".format(args.cohort, args.samp_cutoff),
                     "label-stability_{}.svg".format(args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        "Plots the structure across the inferred similarities of pairs of "
        "mutations tested in a given experiment."
        )

    parser.add_argument('cohort', help='a TCGA cohort', type=str)
    parser.add_argument('samp_cutoff', help='a mutation frequency cutoff',
                        type=int)
    parser.add_argument('classif', help='a mutation classifier', type=str)

    # parse command line arguments, create directory where plots will be saved
    args = parser.parse_args()
    out_tag = "{}__samps-{}".format(args.cohort, args.samp_cutoff)
    os.makedirs(os.path.join(plot_dir, out_tag), exist_ok=True)

    # load inferred mutation relationship metrics generated by the experiment
    with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                  "out-data__{}.p.gz".format(args.classif)),
                     'r') as f:
        infer_df = pickle.load(f)['Infer']

    # load inferred mutation relationship metrics generated by the experiment
    with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                  "out-simil__{}.p.gz".format(args.classif)),
                     'r') as f:
        stat_dict, auc_dict, mutex_dict, siml_dict = pickle.load(f)

    plot_label_stability(infer_df, stat_dict, auc_dict, args)


if __name__ == '__main__':
    main()

