
from ..utilities.mutations import RandomType, ExMcomb, shal_mtype, copy_mtype
from ..subgrouping_isolate import base_dir

import os
import argparse
from pathlib import Path
import bz2
import dill as pickle

import numpy as np
import pandas as pd

from functools import reduce
from operator import and_

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

mpl.use('Agg')
plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'
plot_dir = os.path.join(base_dir, 'plots', 'cohort')


def plot_classif_performance(auc_dicts, time_dicts, cdata, args):
    use_mtypes = reduce(
        and_, [auc_dfs['All'].index[[not isinstance(mtype, RandomType)
                                     for mtype in auc_dfs['All'].index]]
               for auc_dfs in auc_dicts.values()]
        )

    plt_times = pd.Series({
        (clf, ex_lbl): (
            time_df.loc[use_mtypes, 'avg'].apply(np.mean).values
            + time_df.loc[use_mtypes, 'std'].apply(np.mean).values
            ).mean()
        for clf, time_dfs in time_dicts.items()
        for ex_lbl, time_df in time_dfs.items()
        })

    plt_times.index.names = ['Classif', 'Ex']
    clf_times = plt_times.groupby('Classif').mean()
    ex_times = plt_times.groupby('Ex').mean()

    auc_df = pd.concat(
        [pd.DataFrame({'AUC': auc_df.loc[use_mtypes, 'mean'],
                       'Classif': clf, 'Ex': ex_lbl})
         for clf, auc_dfs in auc_dicts.items()
         for ex_lbl, auc_df in auc_dfs.items()]
        )

    fig, ax = plt.subplots(figsize=(1 + 3.1 * len(clf_times), 8))
    sns.violinplot(x='Ex', y='AUC', hue='Classif', data=auc_df,
                   order=['All', 'Iso', 'IsoShal'],
                   hue_order=clf_times.sort_values().index,
                   ax=ax, width=0.89, cut=0)

    for i in range(len(plt_times)):
        ax.get_children()[i * 2].set_alpha(0.71)

    ax.set_xticklabels(["{}\n({:.1f}s)".format(ex_lbl, ex_times[ex_lbl])
                        for ex_lbl in ['All', 'Iso', 'IsoShal']], size=23)
    ax.tick_params(axis='y', labelsize=21)

    ax.axhline(y=0.5, c='black', linewidth=2.7, linestyle=':', alpha=0.71)
    ax.axhline(y=1, c='black', linewidth=2.3, alpha=0.89)
    ax.set_xlabel('')
    ax.set_ylabel('AUC', size=29, weight='semibold')

    fig.savefig(os.path.join(plot_dir, args.expr_source,
                             "{}__classif-performance.svg".format(
                                 args.cohort)),
                bbox_inches='tight', format='svg')

    plt.close()


def plot_projection_dispersion(pred_dicts, auc_dicts, pheno_dict,
                               cdata, args):
    fig, axarr = plt.subplots(figsize=(11, 19), nrows=3)

    use_mcombs = reduce(
        and_, [auc_dfs['All'].index[[isinstance(mtype, ExMcomb)
                                     and not mtype.not_mtype is None
                                     for mtype in auc_dfs['All'].index]]
               for auc_dfs in auc_dicts.values()]
        )

    mtype_dict = {
        'Iso': {mcomb for mcomb in use_mcombs
                if not (mcomb.all_mtype & shal_mtype).is_empty()},
        'IsoShal': {mcomb for mcomb in use_mcombs
                    if (mcomb.all_mtype & shal_mtype).is_empty()}
        }

    lgnd_marks = []
    clf_clrs = sns.color_palette('bright', n_colors=len(pred_dicts))

    for i, (clf, pred_dfs) in enumerate(pred_dicts.items()):
        for ex_lbl, mcombs in mtype_dict.items():
            for mcomb in mcombs:
                auc_val = auc_dicts[clf][ex_lbl].loc[mcomb, 'mean']
                iso_phn = np.array(cdata.train_pheno(mcomb.not_mtype))

                wt_var = pred_dfs[ex_lbl].loc[
                    mcomb, ~iso_phn & ~pheno_dict[mcomb]].apply(np.mean).std()
                mut_var = pred_dfs[ex_lbl].loc[
                    mcomb, pheno_dict[mcomb]].apply(np.mean).std()
                iso_var = pred_dfs[ex_lbl].loc[
                    mcomb, iso_phn].apply(np.mean).std()

                for ax, var_val in zip(axarr, [wt_var, mut_var, iso_var]):
                    ax.scatter(auc_val, var_val, c=[clf_clrs[i]],
                               s=401 * pheno_dict[mcomb].mean(),
                               alpha=0.17, edgecolor='none')

        lgnd_marks += [Line2D([], [], marker='o', linestyle='None',
                              markersize=23, alpha=0.43,
                              markerfacecolor=clf_clrs[i],
                              markeredgecolor='none')]

    fig.legend(lgnd_marks, list(pred_dicts), bbox_to_anchor=(0.5, 0),
               frameon=False, fontsize=19, ncol=3, loc=9, handletextpad=0.19)

    plt.tight_layout(h_pad=1.7)
    fig.savefig(os.path.join(plot_dir, args.expr_source,
                             "{}__proj-dispersion.svg".format(
                                 args.cohort)),
                bbox_inches='tight', format='svg')

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        'plot_cohort',
        description="Plots results across all classifiers in a cohort."
        )

    parser.add_argument('expr_source',
                        help="a source of expression data", type=str)
    parser.add_argument('cohort', help="a TCGA cohort", type=str)

    args = parser.parse_args()
    out_dir = Path(base_dir, '__'.join([args.expr_source, args.cohort]))
    out_list = tuple(out_dir.glob("out-conf__*__*__*.p.gz"))

    if len(out_list) == 0:
        raise ValueError("No completed experiments found for this "
                         "combination of parameters!")

    out_use = pd.DataFrame(
        [{'Levels': '__'.join(out_file.parts[-1].split('__')[1:-2]),
          'Classif': out_file.parts[-1].split('__')[-1].split('.p.gz')[0],
          'File': out_file}
         for out_file in out_list]
        )

    os.makedirs(os.path.join(plot_dir, args.expr_source), exist_ok=True)
    np.random.seed(9087)

    out_iter = out_use.groupby(['Classif', 'Levels'])['File']
    cdata = None
    use_clfs = out_use.Classif.unique()
    phn_dict = dict()

    pred_dicts = {clf: {ex_lbl: pd.DataFrame([])
                        for ex_lbl in ['All', 'Iso', 'IsoShal']}
                  for clf in use_clfs}
    time_dicts = {clf: {ex_lbl: pd.DataFrame([])
                        for ex_lbl in ['All', 'Iso', 'IsoShal']}
                  for clf in use_clfs}
    auc_dicts = {clf: {ex_lbl: pd.DataFrame([])
                       for ex_lbl in ['All', 'Iso', 'IsoShal']}
                 for clf in use_clfs}

    for (clf, lvls), out_files in out_iter:
        out_pred = list()
        out_time = list()
        out_aucs = list()

        for out_file in out_files:
            out_tag = '__'.join(out_file.parts[-1].split('__')[1:])

            with bz2.BZ2File(Path(out_dir,
                                  '__'.join(["cohort-data", out_tag])),
                             'r') as f:
                new_cdata = pickle.load(f)

            if cdata is None:
                cdata = new_cdata
            else:
                cdata.merge(new_cdata)

            with bz2.BZ2File(Path(out_dir, '__'.join(["out-pheno", out_tag])),
                             'r') as f:
                phn_dict.update(pickle.load(f))

            with bz2.BZ2File(Path(out_dir, '__'.join(["out-pred", out_tag])),
                             'r') as f:
                out_pred += [pickle.load(f)]

            with bz2.BZ2File(Path(out_dir, '__'.join(["out-tune", out_tag])),
                             'r') as f:
                out_time += [pickle.load(f)[1]]

            with bz2.BZ2File(Path(out_dir, '__'.join(["out-aucs", out_tag])),
                             'r') as f:
                out_aucs += [pickle.load(f)]

        mtypes_comp = np.greater_equal.outer(
            *([[set(auc_vals['All']['mean'].index)
                for auc_vals in out_aucs]] * 2)
            )
        super_list = np.apply_along_axis(all, 1, mtypes_comp)

        if super_list.any():
            super_indx = super_list.argmax()

            for ex_lbl in ['All', 'Iso', 'IsoShal']:
                pred_dicts[clf][ex_lbl] = pd.concat([
                    pred_dicts[clf][ex_lbl], out_pred[super_indx][ex_lbl]])
                time_dicts[clf][ex_lbl] = pd.concat([
                    time_dicts[clf][ex_lbl], out_time[super_indx][ex_lbl]])
                auc_dicts[clf][ex_lbl] = pd.concat([
                    auc_dicts[clf][ex_lbl], out_aucs[super_indx][ex_lbl]])

    pred_dicts = {clf: {ex_lbl: pred_df.loc[~pred_df.index.duplicated()]
                        for ex_lbl, pred_df in pred_dfs.items()}
                  for clf, pred_dfs in pred_dicts.items()}
    time_dicts = {clf: {ex_lbl: time_df.loc[~time_df.index.duplicated()]
                        for ex_lbl, time_df in time_dfs.items()}
                  for clf, time_dfs in time_dicts.items()}
    auc_dicts = {clf: {ex_lbl: auc_df.loc[~auc_df.index.duplicated()]
                       for ex_lbl, auc_df in auc_dfs.items()}
                 for clf, auc_dfs in auc_dicts.items()}

    plot_classif_performance(auc_dicts, time_dicts, cdata, args)
    plot_projection_dispersion(pred_dicts, auc_dicts, phn_dict, cdata, args)


if __name__ == "__main__":
    main()

