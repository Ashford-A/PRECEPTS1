
"""
Creates plots for baseline testing.
"""

base_dir = '/home/users/grzadkow/compbio/scripts/HetMan/experiments/baseline'
from .config import clf_list, mtype_list
from ..utils import load_output, get_set_plotlbl

import numpy as np
import pandas as pd
from re import sub as gsub

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches


def plot_performance(clf_set='base', mtype_set='default'):
    """Plots barplots of classifier performance for a set of mutations."""
    out_data = load_output('baseline', clf_set, mtype_set)
    alg_order = [clf.__name__ for clf in clf_list[clf_set]]

    # gets AUC data, sets up plot and subplots
    auc_data = [x['AUC'] for x in out_data]
    auc_min = min([min(x.values()) for x in auc_data]) * 0.9
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(7,11))

    for i, gene in enumerate(mtype_list[mtype_set]):

        # cast performance data into matrix format
        perf_data = pd.DataFrame(
            [{k[0].split('_')[0]:v for k,v in x.items() if k[1] == gene}
             for x in auc_data])
        alg_indx = [list(perf_data.columns).index(x) for x in alg_order]
        perf_data = perf_data.ix[:, alg_indx]

        # create and plot the subplot titles describing mutation types
        gene_lbl = '{}-{}'.format(
            gene[0], gsub('(-|, )', '\n', str(gene[1])))
        axes[i // 3, i % 3].set_title(gene_lbl, fontsize=13)

        # plot the boxes showing performances
        axes[i // 3, i % 3].boxplot(
            x=np.array(perf_data),
            boxprops={'linewidth': 1.5},
            medianprops={'linewidth': 3, 'color': '#960c20'},
            flierprops={'markersize': 2}
            )

        # label x-axis ticks with algorithm names if we are on bottom row
        if (i // 3) == 1:
            axes[i // 3, i % 3].set_xticklabels(
                perf_data.columns,
                fontsize=12, rotation=45, ha='right')
        else:
            axes[i // 3, i % 3].set_xticklabels(
                np.repeat('', len(alg_indx)))

        # add y-axis title if we are on left-most column
        if (i % 3) == 0:
            axes[i // 3, i % 3].set_ylabel('AUC', fontsize=19)
        else:
            axes[i // 3, i % 3].set_yticklabels([])

        # add dotted line at AUC=0.5, set AUC axis limits
        axes[i // 3, i % 3].plot(
            list(range(len(alg_indx)+2)), np.repeat(0.5, len(alg_indx)+2),
            c="black", lw=0.8, ls='--', alpha=0.8)
        axes[i // 3, i % 3].set_ylim(auc_min, 1.0)

    # tweak subplot spacing and save plot to file
    plt.tight_layout(w_pad=-1.2, h_pad=1.5)
    plt.savefig(base_dir + '/plots/'
                + get_set_plotlbl(clf_set) + '_' + get_set_plotlbl(mtype_set)
                + '__performance.png',
                dpi=700)


def plot_base(label):
    in_data = load_files(in_dir, 'base_' + label + '_')
    mut_genes = list(list(in_data[0]['AUC'].values())[0].keys())
    clf_list = list(in_data[0]['AUC'].keys())
    neighb_list = list(list(list(
        in_data[0]['AUC'].values())[0].values())[0].keys())
    scores = {gn:None for gn in mut_genes}
    times = {gn:None for gn in mut_genes}

    for mut_gn in mut_genes:
        auc_lower_q = pd.DataFrame(index=clf_list, columns=neighb_list)
        auc_med_q = pd.DataFrame(index=clf_list, columns=neighb_list)
        auc_upper_q = pd.DataFrame(index=clf_list, columns=neighb_list)
        tm_lower_q = pd.DataFrame(index=clf_list, columns=neighb_list)
        tm_med_q = pd.DataFrame(index=clf_list, columns=neighb_list)
        tm_upper_q = pd.DataFrame(index=clf_list, columns=neighb_list)
        scores[mut_gn] = auc_med_q
        times[mut_gn] = tm_med_q

        for neighb in neighb_list:
            for clf in clf_list:
                auc_list = [dt['AUC'][clf][mut_gn][neighb] for dt in in_data]
                tm_list = [dt['time'][clf][mut_gn][neighb] for dt in in_data]
                auc_lower_q.loc[clf, neighb] = np.percentile(auc_list, 25)
                auc_med_q.loc[clf, neighb] = np.percentile(auc_list, 50)
                auc_upper_q.loc[clf, neighb] = np.percentile(auc_list, 75)
                tm_lower_q.loc[clf, neighb] = log(np.percentile(tm_list, 25),
                                                  10)
                tm_med_q.loc[clf, neighb] = log(np.percentile(tm_list, 50),
                                                10)
                tm_upper_q.loc[clf, neighb] = log(np.percentile(tm_list, 75),
                                                  10)

            plt.scatter(
                x=tm_med_q.loc[:, neighb], y=auc_med_q.loc[:, neighb],
                c=tuple(range(len(clf_list))), marker=marker_map[neighb],
                s=55, alpha=0.7)
            plt.legend()

        #new_patch = mpatches.Patch(color=tuple(range(len(clf_list))),
        #                           label=clf_list)
        #plt.legend(handles=[new_patch])
        plt.savefig(out_dir + 'base_' + label + '_' + mut_gn + '.png',
                    bbox_inches='tight')
        plt.clf()

    return scores, times


