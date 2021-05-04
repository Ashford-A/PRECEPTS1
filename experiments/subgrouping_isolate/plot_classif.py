
from ..subgrouping_isolate import base_dir, train_cohorts

import os
import argparse
from pathlib import Path
import bz2
import dill as pickle

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

mpl.use('Agg')
plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'
plot_dir = os.path.join(base_dir, 'plots', 'classif')


def plot_auc_comparisons(auc_dict, pheno_dicts, args):
    fig, ax = plt.subplots(figsize=(6, 7))

    plt_df = pd.DataFrame({ex_lbl: auc_df['mean']
                           for ex_lbl, auc_df in auc_dict.items()}).melt()

    sns.violinplot(x='variable', y='value', data=plt_df, ax=ax, width=0.89,
                   order=['All', 'Iso', 'IsoShal'], cut=0)

    for i in range(3):
        ax.get_children()[i * 2].set_alpha(0.71)

    ax.axhline(y=0.5, c='black', linewidth=2.7, linestyle=':', alpha=0.71)
    ax.axhline(y=1, c='black', linewidth=2.3, alpha=0.89)

    ax.tick_params(axis='x', labelsize=22)
    ax.tick_params(axis='y', labelsize=19)
    ax.set_xlabel('Classification Setup', size=27, weight='semibold')
    ax.set_ylabel('AUC', size=31, weight='semibold')

    fig.savefig(os.path.join(plot_dir,
                             "{}__classif_performance.svg".format(
                                 args.classif)),
                bbox_inches='tight', format='svg')

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        'plot_point',
        description="Compares point mutation subgroupings with a cohort."
        )

    parser.add_argument('classif', help="a mutation classifier")
    parser.add_argument('--data_cache', type=Path)
    args = parser.parse_args()

    if not args.data_cache or not Path.exists(args.data_cache):
        out_datas = tuple(Path(base_dir).glob(
            os.path.join("*", "out-aucs__*__semideep__{}.p.gz".format(
                args.classif))
            ))

        out_list = pd.DataFrame(
            [{'Source': '__'.join(out_data.parts[-2].split('__')[:-1]),
              'Cohort': out_data.parts[-2].split('__')[-1],
              'Levels': '__'.join(out_data.parts[-1].split('__')[1:-2]),
              'File': out_data}
             for out_data in out_datas]
            ).groupby('Cohort').filter(
                lambda outs: 'Consequence__Exon' in set(outs.Levels))

        if len(out_list) == 0:
            raise ValueError("No completed experiments found for this "
                             "combination of parameters!")

        out_list = out_list[out_list.Cohort.isin(train_cohorts)]
        use_iter = out_list.groupby(['Source', 'Cohort', 'Levels'])['File']

        out_dirs = {(src, coh): Path(base_dir, '__'.join([src, coh]))
                    for src, coh, _ in use_iter.groups}
        out_tags = {fl: '__'.join(fl.parts[-1].split('__')[1:])
                    for fl in out_list.File}

        phn_dicts = {(src, coh): dict() for src, coh, _ in use_iter.groups}
        auc_dfs = {(src, coh): {ex_lbl: pd.DataFrame()
                                for ex_lbl in ['All', 'Iso', 'IsoShal']}
                   for src, coh, _ in use_iter.groups}

        for (src, coh, lvls), out_files in use_iter:
            out_aucs = list()

            for out_file in out_files:
                with bz2.BZ2File(Path(out_dirs[src, coh],
                                      '__'.join(["out-pheno",
                                                 out_tags[out_file]])),
                                 'r') as f:
                    phn_dicts[src, coh].update(pickle.load(f))

                with bz2.BZ2File(Path(out_dirs[src, coh],
                                      '__'.join(["out-aucs",
                                                 out_tags[out_file]])),
                                 'r') as f:
                    out_aucs += [pickle.load(f)]

            mtypes_comp = np.greater_equal.outer(
                *([[set(auc_dict['All'].index)
                    for auc_dict in out_aucs]] * 2)
                )
            super_comp = np.apply_along_axis(all, 1, mtypes_comp)

            # if there is not a subgrouping set that contains all the others,
            # concatenate the output of all sets...
            if not super_comp.any():
                for ex_lbl in ['All', 'Iso', 'IsoShal']:
                    auc_dfs[src, coh][ex_lbl] = auc_dfs[
                        src, coh][ex_lbl].append(
                            pd.concat([aucs[ex_lbl] for aucs in out_aucs]))

            # ...otherwise, use the "superset"
            else:
                super_indx = super_comp.argmax()

                for ex_lbl in ['All', 'Iso', 'IsoShal']:
                    auc_dfs[src, coh][ex_lbl] = auc_dfs[
                        src, coh][ex_lbl].append(out_aucs[super_indx][ex_lbl])

        # filter out duplicate subgroupings due to overlapping search criteria
        for src, coh, _ in use_iter.groups:
            for ex_lbl in ['All', 'Iso', 'IsoShal']:
                auc_dfs[src, coh][ex_lbl].sort_index(inplace=True)
                auc_dfs[src, coh][ex_lbl] = auc_dfs[src, coh][ex_lbl].loc[
                    ~auc_dfs[src, coh][ex_lbl].index.duplicated()]

        auc_dict = dict()
        for ex_lbl in ['All', 'Iso', 'IsoShal']:
            auc_dict[ex_lbl] = pd.DataFrame({
                (src, coh, mut): auc_vals
                for (src, coh), auc_df in auc_dfs.items()
                for mut, auc_vals in auc_df[ex_lbl].iterrows()
                }).transpose()

            auc_dict[ex_lbl]['mean'] = auc_dict[ex_lbl]['mean'].astype(float)
            auc_dict[ex_lbl]['all'] = auc_dict[ex_lbl]['all'].astype(float)

        if args.data_cache:
            with bz2.BZ2File(args.data_cache, 'w') as f:
                pickle.dump((phn_dicts, auc_dict), f, protocol=-1)

    else:
        with bz2.BZ2File(args.data_cache, 'r') as f:
            phn_dicts, auc_dict = pickle.load(f)

    os.makedirs(plot_dir, exist_ok=True)
    plot_auc_comparisons(auc_dict, phn_dicts, args)


if __name__ == '__main__':
    main()

