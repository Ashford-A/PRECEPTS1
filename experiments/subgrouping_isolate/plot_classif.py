
from ..utilities.mutations import (pnt_mtype, copy_mtype, dels_mtype,
                                   Mcomb, ExMcomb)
from dryadic.features.mutations import MuType

from ..subgrouping_isolate import base_dir, train_cohorts
from .utils import get_mut_ex, get_mcomb_lbl
from ..utilities.labels import get_fancy_label, get_cohort_label
from ..utilities.label_placement import place_scatter_labels
from ..subgrouping_test.plot_aucs import add_scatterpie_legend
from ..utilities.misc import get_label, get_subtype, choose_label_colour

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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

mpl.use('Agg')
plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'
plot_dir = os.path.join(base_dir, 'plots', 'classif')


sub_filters = {
    'Point': (lambda mut: ((isinstance(mut, (Mcomb, ExMcomb))
                            and len(mut.mtypes) == 1
                            and any((get_subtype(mtype)
                                     & copy_mtype).is_empty()
                                    for mtype in mut.mtypes))
                           or (not isinstance(mut, (Mcomb, ExMcomb))
                               and (get_subtype(mut)
                                    & copy_mtype).is_empty())),
              pnt_mtype),

    'CopyPoint': (
        lambda mut: (isinstance(mut, ExMcomb) and len(mut.mtypes) == 2
                     and sum((get_subtype(mtype) & copy_mtype).is_empty()
                             for mtype in mut.mtypes) == 1
                     and any(get_subtype(mtype) == pnt_mtype
                             for mtype in mut.mtypes)),
        pnt_mtype
        ),

    'CopyLoss': (
        lambda mut: (isinstance(mut, ExMcomb) and len(mut.mtypes) == 2
                     and sum((get_subtype(mtype) & dels_mtype).is_empty()
                             for mtype in mut.mtypes) == 1
                     and any(get_subtype(mtype) == pnt_mtype
                             for mtype in mut.mtypes)),
        pnt_mtype
        ),

    'Points': (
        lambda mut: (isinstance(mut, ExMcomb) and len(mut.mtypes) == 2
                     and all((get_subtype(mtype) & copy_mtype).is_empty()
                             for mtype in mut.mtypes)),
        pnt_mtype
        ),
    }


def plot_auc_comparisons(auc_dict, pheno_dicts, args):
    fig, ax = plt.subplots(figsize=(4, 7))

    auc_df = auc_dict['All']
    auc_df = auc_df.loc[[
        (isinstance(mut, (Mcomb, ExMcomb))
         and all((get_subtype(mtype) & copy_mtype).is_empty()
                 or (get_subtype(mtype) & pnt_mtype).is_empty()
                 for mtype in mut.mtypes))

        or (not isinstance(mut, (Mcomb, ExMcomb))
            and ((get_subtype(mut) & copy_mtype).is_empty()
                 or (get_subtype(mut) & pnt_mtype).is_empty()))

        for _, _, mut in auc_df.index
        ]]

    iso_df = auc_df.loc[[isinstance(mut, ExMcomb) and get_mut_ex(mut) == 'Iso'
                         for _, _, mut in auc_df.index]]
    use_indx = list()

    for src, coh, mcomb in iso_df.index:
        if len(mcomb.mtypes) == 1:
            use_mtype = tuple(mcomb.mtypes)[0]
        else:
            use_mtype = Mcomb(*mcomb.mtypes)

        use_indx += [(src, coh, use_mtype)]

    plt_df = pd.DataFrame({
        'All': auc_dict['All'].loc[use_indx, 'mean'].values,
        'Iso': iso_df['mean'].values
        }).melt()

    sns.violinplot(x='variable', y='value', data=plt_df,
                   ax=ax, width=0.91, order=['All', 'Iso'],
                   palette=['0.17', "#3258F3"], cut=0)

    for i in range(2):
        vio_clr = ax.get_children()[i * 2].get_facecolor()
        vio_clr[0, 3] = 0.61
        ax.get_children()[i * 2].set_facecolor(vio_clr)

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


def plot_sub_comparisons(auc_df, pheno_dicts,
                         args, ex_lbl, use_filter, all_aucs, add_lgnd=True):
    fig, ax = plt.subplots(figsize=(10.3, 11))

    filter_fx, base_subtype = sub_filters[use_filter]
    use_aucs = auc_df[[get_mut_ex(mut) == ex_lbl and filter_fx(mut)
                       for _, _, mut in auc_df.index]]['mean']

    plot_dict = dict()
    line_dict = dict()
    plt_min = 0.57

    for (src, coh, gene), auc_vec in use_aucs.groupby(
            lambda x: (x[0], x[1], get_label(x[2]))):
        base_mtype = MuType({('Gene', gene): base_subtype})
        sub_aucs = auc_vec[[mut != base_mtype for _, _, mut in auc_vec.index]]

        if len(sub_aucs) == 0:
            continue

        best_subtype = sub_aucs.idxmax()[2]
        auc_tupl = (auc_df.loc[(src, coh, base_mtype), 'mean'],
                    auc_vec[src, coh, best_subtype])
        plt_min = min(plt_min, auc_tupl[0] - 0.03, auc_tupl[1] - 0.029)

        base_size = np.mean(pheno_dicts[src, coh][base_mtype])
        best_prop = np.mean(pheno_dicts[src, coh][best_subtype])
        best_prop /= base_size
        plt_size = 0.07 * base_size ** 0.5
        plot_dict[auc_tupl] = [plt_size, ('', '')]
        line_dict[auc_tupl] = dict(c=choose_label_colour(gene))

        cv_sig = (np.array(auc_df['CV'][src, coh, best_subtype])
                  > np.array(auc_df['CV'][src, coh, base_mtype])).all()

        # ...and if we are sure that the optimal subgrouping AUC is
        # better than the point mutation AUC then add a label with the
        # gene name and a description of the best found subgrouping...
        if auc_vec.max() >= 0.7:
            gene_lbl = "{} in {}".format(gene, get_cohort_label(coh))

            if cv_sig:
                if isinstance(best_subtype, MuType):
                    plot_dict[auc_tupl][1] = (
                        gene_lbl,
                        get_fancy_label(get_subtype(best_subtype),
                                        pnt_link='\nor ', phrase_link=' ')
                        )

                else:
                    plot_dict[auc_tupl][1] = (gene_lbl,
                                              get_mcomb_lbl(best_subtype))

        pie_bbox = (auc_tupl[0] - plt_size / 2,
                    auc_tupl[1] - plt_size / 2, plt_size, plt_size)

        pie_ax = inset_axes(ax, width='100%', height='100%',
                            bbox_to_anchor=pie_bbox,
                            bbox_transform=ax.transData,
                            axes_kwargs=dict(aspect='equal'), borderpad=0)

        pie_ax.pie(x=[best_prop, 1 - best_prop],
                   colors=[line_dict[auc_tupl]['c'] + (0.77,),
                           line_dict[auc_tupl]['c'] + (0.29,)],
                   explode=[0.29, 0], startangle=90)

    plt_lims = plt_min, 1 + (1 - plt_min) / 181
    ax.grid(linewidth=0.83, alpha=0.41)

    ax.plot(plt_lims, [0.5, 0.5],
            color='black', linewidth=1.3, linestyle=':', alpha=0.71)
    ax.plot([0.5, 0.5], plt_lims,
            color='black', linewidth=1.3, linestyle=':', alpha=0.71)

    ax.plot(plt_lims, [1, 1], color='black', linewidth=1.9, alpha=0.89)
    ax.plot([1, 1], plt_lims, color='black', linewidth=1.9, alpha=0.89)
    ax.plot(plt_lims, plt_lims,
            color='#550000', linewidth=2.1, linestyle='--', alpha=0.41)

    ax.set_xlabel("Accuracy of Gene-Wide Classifier",
                  size=23, weight='semibold')
    ax.set_ylabel("Accuracy of Best Subgrouping Classifier",
                  size=23, weight='semibold')

    if add_lgnd:
        ax, plot_dict = add_scatterpie_legend(ax, plot_dict, plt_min,
                                              pnt_mtype, args)

    if plot_dict:
        lbl_pos = place_scatter_labels(plot_dict, ax,
                                       plt_lims=[[plt_min + 0.01, 0.99]] * 2,
                                       line_dict=line_dict)

    ax.set_xlim(plt_lims)
    ax.set_ylim(plt_lims)

    plt.savefig(os.path.join(plot_dir,
                             "{}-sub{}-comparisons_{}.svg".format(
                                 ex_lbl, use_filter, args.classif)),
                bbox_inches='tight', format='svg')

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        'plot_point',
        description="Compares point mutation subgroupings with a cohort."
        )

    parser.add_argument('classif', help="a mutation classifier")
    parser.add_argument('--data_cache', type=Path)
    parser.add_argument('--filters', nargs='+', default=['Point'])
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

    for filter_lbl in args.filters:
        cv_dict = dict()
        acc_dict = dict()
        filter_fx, base_subtype = sub_filters[filter_lbl]

        use_aucs = auc_dict['Iso'][[
            get_mut_ex(mut) == 'Iso' and filter_fx(mut)
            for _, _, mut in auc_dict['Iso'].index
            ]]['mean']

        for (src, coh, gene), auc_vec in use_aucs.groupby(
                lambda x: (x[0], x[1], get_label(x[2]))):
            base_mtype = MuType({('Gene', gene): base_subtype})
            sub_aucs = auc_vec[[mut != base_mtype
                                for _, _, mut in auc_vec.index]]

            if len(sub_aucs) == 0:
                cv_dict[coh, gene] = -1

            else:
                cv_dict[coh, gene] = max(
                    (np.array(auc_dict['Iso']['CV'][comb])
                     > np.array(auc_dict['All']['CV'][src, coh, base_mtype])
                     ).sum()
                    for comb in sub_aucs.index
                    )

                acc_dict[coh, gene, base_mtype] = auc_dict['All']['mean'][
                    src, coh, base_mtype]
                for comb in sub_aucs.index:
                    acc_dict[coh, gene, comb] = auc_dict['Iso']['mean'][comb]

    for ex_lbl, auc_df in auc_dict.items():
        for filter_lbl in args.filters:
            plot_sub_comparisons(auc_df, phn_dicts, args, ex_lbl, filter_lbl,
                                 auc_dict['All'])


if __name__ == '__main__':
    main()

