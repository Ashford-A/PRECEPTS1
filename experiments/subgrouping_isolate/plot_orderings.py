
import os
import sys

base_dir = os.path.join(os.environ['DATADIR'], 'HetMan',
                        'subgrouping_isolate')
sys.path.extend([os.path.join(os.path.dirname(__file__), '..', '..', '..')])
plot_dir = os.path.join(base_dir, 'plots', 'orderings')

from HetMan.experiments.utilities.mutations import ExMcomb
from HetMan.experiments.subvariant_isolate import cna_mtypes
from HetMan.experiments.subvariant_test import pnt_mtype, copy_mtype
from dryadic.features.mutations import MuType

from HetMan.experiments.subgrouping_isolate.utils import calculate_pair_siml
from HetMan.experiments.subvariant_isolate.utils import get_fancy_label
from HetMan.experiments.utilities.colour_maps import simil_cmap

import argparse
from pathlib import Path
import bz2
import dill as pickle

import numpy as np
import pandas as pd

from itertools import product
from scipy.spatial import distance
from scipy.cluster.hierarchy import linkage, dendrogram

import matplotlib as mpl
mpl.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase
from matplotlib import colors


def get_xaxis_labels(mcombs, pheno_dict):
    return [
        "{} ({})".format(
            ' & '.join([get_fancy_label(mtype.subtype_list()[0][1],
                                        phrase_link=' ')
                        for mtype in mcomb.mtypes]),
            np.sum(pheno_dict[mcomb])
            )
        for mcomb in mcombs
        ]


def get_yaxis_labels(mcombs):
    yaxis_lbls = [[str(mtype).split(':')[1:] for mtype in mcomb.mtypes]
                  for mcomb in mcombs]

    yaxis_lbls = [
        ' & '.join([lbl[-1][2:] if len(lbl[-1]) > 2 and lbl[-1][:2] == 'p.'
                    else ':'.join(lbl[1:]) if len(lbl) > 1
                    else ':'.join(lbl) for lbl in ylab])
        for ylab in yaxis_lbls
        ]

    return yaxis_lbls


def plot_singleton_ordering(siml_dicts, auc_vals, pheno_dict, pred_df,
                            ex_lbl, cdata, args, cluster=False):
    use_gene = set(mcomb.get_labels()[0] for mcomb in auc_vals.index)
    assert len(use_gene) == 1, ("This plot can only be created using the "
                                "mutation combinations from one gene!")

    use_gene = tuple(use_gene)[0]
    all_type = MuType(tuple(cdata.mtrees.values())[0][use_gene].allkey())

    if ex_lbl == 'IsoShal':
        all_type -= dict(cna_mtypes)['Shal']
    all_mtype = MuType({('Gene', use_gene): all_type})

    singl_mcombs = {
        mcomb for mcomb, auc_val in auc_vals.iteritems()
        if (auc_val >= 0.6 and all(len(mtype.subkeys()) == 1
                                   or (mtype & pnt_mtype).is_empty()
                                   for mtype in mcomb.mtypes))
        }

    if ex_lbl == 'IsoShal':
        singl_mcombs = {mcomb for mcomb in singl_mcombs
                        if (tuple(mcomb.mtypes)[0]
                            & dict(cna_mtypes)['Shal']).is_empty()}

    if len(singl_mcombs) <= 1:
        return None

    fig_size = 5. + len(singl_mcombs) * 0.43
    fig, (heat_ax, lgnd_ax) = plt.subplots(
        figsize=(fig_size, fig_size), nrows=1, ncols=2,
        gridspec_kw=dict(width_ratios=[4 + len(singl_mcombs) * 0.43, 1])
        )

    siml_df = pd.DataFrame(index=singl_mcombs, columns=singl_mcombs,
                           dtype=float)

    if args.test:
        test_list = list()
    else:
        test_list = None

    for mcomb1, mcomb2 in product(singl_mcombs, repeat=2):
        if args.test:
            siml_df.loc[mcomb1, mcomb2], test_list = calculate_pair_siml(
                mcomb1, mcomb2, all_mtype, siml_dicts,
                pheno_dict, pred_df, cdata, test_list
                )

        else:
            siml_df.loc[mcomb1, mcomb2] = calculate_pair_siml(
                mcomb1, mcomb2, all_mtype, siml_dicts,
                pheno_dict, pred_df, cdata, test_list
                )

    if args.test:
        print("Successfully tested the copy-similarities of "
              "{} mutation pairs within {} for internal consistency!".format(
                  len(test_list), use_gene))

    #TODO: place the dendrogram on top of the heatmap?
    if cluster:
        plt_tag = 'singleton-clust'
        siml_order = siml_df.index[
            dendrogram(linkage(distance.pdist(siml_df, metric='cityblock'),
                               method='centroid'), no_plot=True)['leaves']
            ]

    else:
        plt_tag = 'singleton-order'
        siml_rank = siml_df.mean(axis=1) - siml_df.mean(axis=0)
        siml_order = siml_rank.sort_values().index

    siml_df = siml_df.loc[siml_order, siml_order]
    annot_df = siml_df.copy()
    annot_df[annot_df < 3.] = 0.0
    for mcomb in singl_mcombs:
        annot_df.loc[mcomb, mcomb] = auc_vals[mcomb]

    annot_df = annot_df.applymap('{:.2f}'.format).applymap(
        lambda x: ('' if x == '0.00' else '1.0' if x == '1.00'
                   else x.lstrip('0'))
        )

    # draw the heatmap
    sns.heatmap(siml_df, cmap=simil_cmap, cbar=False,
                vmin=-1., vmax=2., ax=heat_ax, square=True,
                annot=annot_df, fmt='', annot_kws={'size': 14})

    xlabs = get_xaxis_labels(siml_df.index, pheno_dict)
    ylabs = get_yaxis_labels(siml_df.columns)

    heat_ax.set_xticklabels(xlabs, rotation=31, ha='right', size=15)
    heat_ax.set_yticklabels(ylabs, size=13)
    heat_ax.set_xlabel("M2: Testing Mutation (# of samples)",
                       size=31, weight='semibold')
    heat_ax.set_ylabel("M1: Training Mutation", size=33, weight='semibold')

    heat_ax.set_xlim(0, len(singl_mcombs))
    heat_ax.set_ylim(0, len(singl_mcombs))

    clr_ax = lgnd_ax.inset_axes(bounds=(-0.47, 0.21, 1.09, 0.58),
                                clip_on=False)

    clr_bar = ColorbarBase(ax=clr_ax, cmap=simil_cmap,
                           norm=colors.Normalize(vmin=-1, vmax=2),
                           extend='both', extendfrac=0.19,
                           ticks=[-0.73, 0, 0.5, 1.0, 1.73])

    clr_bar.ax.set_yticklabels(
        ['M2 < WT', 'M2 = WT', 'WT < M2 < M1', 'M2 = M1', 'M2 > M1'],
        size=23, fontweight='bold'
        )
    lgnd_ax.axis('off')

    plt.savefig(
        os.path.join(plot_dir, '__'.join([args.expr_source, args.cohort]),
                     "{}_{}_{}_{}.svg".format(
                         use_gene, ex_lbl, plt_tag, args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        "Plot the structure of a gene's subtypes in a given cohort based on "
        "how their isolated expression signatures classify one another."
        )

    parser.add_argument('expr_source', help="a source of expression datasets")
    parser.add_argument('cohort', help="a tumour cohort")
    parser.add_argument('classif', help="a mutation classifier")

    parser.add_argument('--genes', '-g', nargs='+',
                        help="restrict plots drawn to these mutated genes?")
    parser.add_argument('--test', action='store_true',
                        help="run diagnostic tests?")

    args = parser.parse_args()
    out_dir = Path(base_dir, '__'.join([args.expr_source, args.cohort]))
    out_list = tuple(Path(out_dir).glob(
        "out-siml__*__*__{}.p.gz".format(args.classif)))

    if len(out_list) == 0:
        raise ValueError("No completed experiments found for this "
                         "combination of parameters!")

    out_use = pd.DataFrame(
        [{'Levels': '__'.join(out_file.parts[-1].split('__')[1:-2]),
          'File': out_file}
         for out_file in out_list]
        )

    if 'Consequence__Exon' not in set(out_use.Levels):
        raise ValueError("Cannot compare AUCs until this experiment is run "
                         "with mutation levels `Consequence__Exon` "
                         "which tests genes' base mutations!")

    os.makedirs(os.path.join(
        plot_dir, '__'.join([args.expr_source, args.cohort])), exist_ok=True)

    out_iter = out_use.groupby('Levels')['File']
    out_aucs = {lvls: list() for lvls in out_iter.groups}
    out_simls = {lvls: list() for lvls in out_iter.groups}
    out_preds = {lvls: list() for lvls in out_iter.groups}

    phn_dict = dict()
    cdata = None

    auc_df = pd.DataFrame([])
    siml_dicts = {ex_lbl: {lvls: None for lvls in out_iter.groups}
                  for ex_lbl in ['Iso', 'IsoShal']}
    pred_dfs = {ex_lbl: pd.DataFrame([]) for ex_lbl in ['Iso', 'IsoShal']}

    for lvls, out_files in out_iter:
        for out_file in out_files:
            out_tag = '__'.join(out_file.parts[-1].split('__')[1:])

            with bz2.BZ2File(Path(out_dir, '__'.join(["out-pheno", out_tag])),
                             'r') as f:
                phn_vals = pickle.load(f)
                phn_vals = {mut: phns for mut, phns in phn_vals.items()
                            if isinstance(mut, ExMcomb)}

                if args.genes:
                    phn_vals = {
                        mcomb: phns for mcomb, phns in phn_vals.items()
                        if mcomb.get_labels()[0] in set(args.genes)
                        }

                phn_dict.update(phn_vals)

            with bz2.BZ2File(Path(out_dir, '__'.join(["out-aucs", out_tag])),
                             'r') as f:
                auc_vals = pickle.load(f)

                auc_vals = pd.DataFrame({
                    ex_lbl: auc_vals[ex_lbl]['mean'][
                        [mut for mut in auc_vals[ex_lbl]['mean'].index
                         if isinstance(mut, ExMcomb)]
                        ]
                    for ex_lbl in ['Iso', 'IsoShal']
                    })

                if args.genes:
                    auc_vals = auc_vals.loc[[
                        mcomb for mcomb in auc_vals.index
                        if mcomb.get_labels()[0] in set(args.genes)
                        ]]

                out_aucs[lvls] += [auc_vals]

            with bz2.BZ2File(Path(out_dir, '__'.join(["out-siml", out_tag])),
                             'r') as f:
                siml_vals = pickle.load(f)

                if args.genes:
                    siml_vals = {
                        ex_lbl: {
                            mcomb1: {
                                mcomb2: siml_val
                                for mcomb2, siml_val in siml_dict.items()
                                if mcomb2.get_labels()[0] in set(args.genes)
                                }
                            for mcomb1, siml_dict in siml_dicts.items()
                            if mcomb1.get_labels()[0] in set(args.genes)
                            }
                        for ex_lbl, siml_dicts in siml_vals.items()
                        }

                out_simls[lvls] += [siml_vals]

            with bz2.BZ2File(Path(out_dir, '__'.join(["out-pred", out_tag])),
                             'r') as f:
                pred_vals = pickle.load(f)

                pred_vals = {
                    ex_lbl: pred_vals[ex_lbl].loc[
                        [mut for mut in pred_vals[ex_lbl].index
                         if isinstance(mut, ExMcomb)]
                        ]
                    for ex_lbl in ['Iso', 'IsoShal']
                    }

                if args.genes:
                    pred_vals = {
                        ex_lbl: pred_mat.loc[
                            [mcomb for mcomb in pred_mat.index
                             if mcomb.get_labels()[0] in set(args.genes)]
                            ]
                        for ex_lbl, pred_mat in pred_vals.items()
                        }

                out_preds[lvls] += [pred_vals]

            with bz2.BZ2File(Path(out_dir,
                                  '__'.join(["cohort-data", out_tag])),
                             'r') as f:
                new_cdata = pickle.load(f)

                if cdata is None:
                    cdata = new_cdata
                else:
                    cdata.merge(new_cdata, use_genes=args.genes)

        mtypes_comp = np.greater_equal.outer(
            *([[set(auc_vals['Iso'].index)
                for auc_vals in out_aucs[lvls]]] * 2)
            )
        super_list = np.apply_along_axis(all, 1, mtypes_comp)

        if super_list.any():
            super_indx = super_list.argmax()

            auc_df = pd.concat([auc_df, out_aucs[lvls][super_indx]],
                               sort=False)

            for ex_lbl in ['Iso', 'IsoShal']:
                siml_dicts[ex_lbl][lvls] = out_simls[lvls][super_indx][ex_lbl]
                pred_dfs[ex_lbl] = pd.concat([
                    pred_dfs[ex_lbl], out_preds[lvls][super_indx][ex_lbl]],
                    sort=False
                    )

    if cdata.muts.shape[0] == 0:
        raise ValueError("No mutation calls found in cohort "
                         "`{}` for these genes!".format(args.cohort))

    if not phn_dict:
        raise ValueError("No mutation types passing test search criteria "
                         "found for this combination of parameters!")

    auc_df = auc_df.loc[~auc_df.index.duplicated()]
    use_mtypes = {
        'Iso': {mcomb for mcomb in auc_df.index
                if not (mcomb.all_mtype
                        & dict(cna_mtypes)['Shal']).is_empty()},
        'IsoShal': {mcomb for mcomb in auc_df.index
                    if (mcomb.all_mtype
                        & dict(cna_mtypes)['Shal']).is_empty()}
        }

    assert not (use_mtypes['Iso'] & use_mtypes['IsoShal'])
    assert (use_mtypes['Iso'] | use_mtypes['IsoShal']) == set(auc_df.index)

    for ex_lbl in ['Iso', 'IsoShal']:
        for gene, auc_vals in auc_df.loc[use_mtypes[ex_lbl], ex_lbl].groupby(
                lambda mcomb: mcomb.get_labels()[0]):
            for mcomb_cluster in [False, True]:
                plot_singleton_ordering(
                    siml_dicts[ex_lbl], auc_vals, phn_dict, pred_dfs[ex_lbl],
                    ex_lbl, cdata, args, cluster=mcomb_cluster
                    )


if __name__ == '__main__':
    main()

