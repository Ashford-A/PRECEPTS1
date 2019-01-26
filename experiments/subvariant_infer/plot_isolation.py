
import os
import sys

if 'DATADIR' in os.environ:
    base_dir = os.path.join(os.environ['DATADIR'],
                            'HetMan', 'subvariant_infer')
else:
    base_dir = os.path.dirname(__file__)

plot_dir = os.path.join(base_dir, 'plots', 'isolation')
sys.path.extend([os.path.join(os.path.dirname(__file__), '../../..')])

from HetMan.experiments.subvariant_infer import ExMcomb
from HetMan.experiments.subvariant_infer.fit_infer import load_cohort_data
from HetMan.experiments.subvariant_infer.utils import load_infer_output
from dryadic.features.mutations import MuType

import argparse
import numpy as np
import pandas as pd
from itertools import product
from operator import itemgetter

import matplotlib as mpl
mpl.use('Agg')

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.collections import PolyCollection

mut_clrs = {'Mutant': sns.light_palette('#C50000', reverse=True)[0],
            'Wild-Type': '0.29'}

variant_mtypes = (
    ('Loss Alterations', MuType({('Scale', 'Copy'): {(
        'Copy', ('ShalDel', 'DeepDel')): None}})),
    ('Other\nPoint Mutations', MuType({('Scale', 'Point'): None})),
    ('Gain Alterations', MuType({('Scale', 'Copy'): {(
        'Copy', ('ShalGain', 'DeepGain')): None}}))
    )


def calc_auc(vals, stat):
    return (np.greater.outer(vals[stat], vals[~stat]).mean()
            + 0.5 * np.equal.outer(vals[stat], vals[~stat]).mean())


def plot_mtype_distributions(mtypes, infer_dict, cdata, args):
    fig, axarr = plt.subplots(figsize=(0.1 + 2.8 * len(mtypes), 9),
                              nrows=2, ncols=len(mtypes))

    gene_stat = np.array(cdata.train_pheno(
        MuType({('Gene', args.gene): None})))
    all_mtype = MuType(cdata.train_mut.allkey())

    for j, cur_mtype in enumerate(mtypes):
        rest_stat = np.array(cdata.train_pheno(all_mtype - cur_mtype))

        infer_df = pd.DataFrame({
            'Value': infer_dict['All'].loc[cur_mtype],
            'cStat': np.array(cdata.train_pheno(cur_mtype)),
            'rStat': np.array(cdata.train_pheno(all_mtype - cur_mtype))
            })

        mtype_str = str(cur_mtype).split(':')[-1]
        axarr[0, j].text(0.5, 1.01,
                         "{}({}) mutations\n({} affected samples)".format(
                             args.gene, mtype_str, np.sum(infer_df.cStat)),
                         size=12, ha='center', va='bottom',
                         transform=axarr[0, j].transAxes)

        vals_min, vals_max = infer_df.Value.quantile(q=[0, 1])
        vals_rng = (vals_max - vals_min) / 31
        axarr[0, j].set_ylim(vals_min - 6 * vals_rng, vals_max + 3 * vals_rng)

        sns.violinplot(data=infer_df[~infer_df.cStat], y='Value',
                       palette=[mut_clrs['Wild-Type']],
                       linewidth=0, cut=0, ax=axarr[0, j])
        sns.violinplot(data=infer_df[infer_df.cStat], y='Value',
                       palette=[mut_clrs['Mutant']],
                       linewidth=0, cut=0, ax=axarr[0, j])

        axarr[0, j].text(0.5, 0.97,
                         "{} AUC: {:.3f}".format(args.classif,
                                                 calc_auc(infer_df.Value,
                                                          infer_df.cStat)),
                         size=10, ha='center', va='top',
                         transform=axarr[0, j].transAxes)

        axarr[0, j].legend(
            [Patch(color=mut_clrs['Mutant'], alpha=0.36),
             Patch(color=mut_clrs['Wild-Type'], alpha=0.36)],
            ["{} Mutants".format(mtype_str),
             "{} Wild-Types".format(mtype_str)],
            fontsize=11, ncol=1, loc=8, bbox_to_anchor=(0.5, -0.01)
            ).get_frame().set_linewidth(0.0)

        sns.violinplot(data=infer_df[~infer_df.cStat], x='cStat', y='Value',
                       hue='rStat', palette=[mut_clrs['Wild-Type']],
                       hue_order=[False, True], split=True, linewidth=0,
                       cut=0, ax=axarr[1, j])
        sns.violinplot(data=infer_df[infer_df.cStat], x='cStat', y='Value',
                       hue='rStat', palette=[mut_clrs['Mutant']],
                       hue_order=[False, True], split=True, linewidth=0,
                       cut=0, ax=axarr[1, j])

        axarr[1, j].get_legend().remove()
        axarr[1, j].set_ylim(vals_min - 2 * vals_rng, vals_max + 2 * vals_rng)
        axarr[1, j].axvline(x=0, ymin=-1, ymax=2,
                            color='black', linewidth=1.1, alpha=0.61)

        axarr[1, j].text(
            0.09, 0.98, "AUC: {:.3f}".format(
                calc_auc(infer_df.Value[~infer_df.rStat],
                         infer_df.cStat[~infer_df.rStat])
                ),
            size=9, ha='left', va='top', transform=axarr[1, j].transAxes
            )

        axarr[1, j].text(
            0.91, 0.98, "AUC: {:.3f}".format(
                calc_auc(infer_df.Value[infer_df.rStat],
                         infer_df.cStat[infer_df.rStat])
                ),
            size=9, ha='right', va='top', transform=axarr[1, j].transAxes
            )

        vio_mesh = (
            ('wtWO', np.stack([[0.45] * 40, np.arange(0.1, 0.5, 0.01)],
                              axis=1)),
            ('mutWO', np.stack([[0.45] * 30, np.arange(0.57, 0.865, 0.01)],
                               axis=1)),
            ('wtW', np.stack([[0.55] * 40, np.arange(0.1, 0.5, 0.01)],
                             axis=1)),
            ('mutW', np.stack([[0.55] * 30, np.arange(0.57, 0.865, 0.01)],
                              axis=1))
            )

        vio_indx = {'wtWO': 0, 'wtW': 1, 'mutWO': 3, 'mutW': 4}
        pos_deflt = {'wtWO': 0.06, 'wtW': 0.06, 'mutWO': 0.88, 'mutW': 0.88}

        lbl_txt = {
            'wtWO': "{} wt w/o\nother {} muts".format(mtype_str, args.gene),
            'mutWO': "{} mut w/o\nother {} muts".format(mtype_str, args.gene),
            'wtW': "{} wt w/\nother {} muts".format(mtype_str, args.gene),
            'mutW': "{} mut w/\nother {} muts".format(mtype_str, args.gene)
            }

        for i in range(2):
            axarr[i, j].set_xticks([])
            axarr[i, j].xaxis.label.set_visible(False)
            axarr[i, j].yaxis.label.set_visible(False)
            axarr[i, j].set_xticklabels([])
            axarr[i, j].set_yticklabels([])

            for art in axarr[i, j].get_children():
                if isinstance(art, PolyCollection):
                    art.set_alpha(0.41)

        for lbl, mesh in vio_mesh:
            vio_ovlp = axarr[1, j].get_children()[
                vio_indx[lbl]].get_paths()[0].contains_points(
                    axarr[1, j].transAxes.transform(mesh),
                    transform=axarr[i, j].transData
                    )

            if np.all(vio_ovlp):
                ypos = pos_deflt[lbl]

            else:
                if lbl[:2] == 'wt':
                    ypos = np.max(mesh[:, 1][~vio_ovlp])
                else:
                    ypos = np.min(mesh[:, 1][~vio_ovlp])

            if lbl[:2] == 'wt':
                str_clr = mut_clrs['Wild-Type']
                str_va = 'top'
            else:
                str_clr = mut_clrs['Mutant']
                str_va = 'bottom'

            if lbl[-2:] == 'WO':
                str_ha = 'right'
            else:
                str_ha = 'left'

            axarr[1, j].text(mesh[0, 0], ypos, lbl_txt[lbl],
                             color=str_clr, size=7, ha=str_ha, va=str_va,
                             transform=axarr[1, j].transAxes)

    plt.tight_layout()
    plt.savefig(os.path.join(
        plot_dir, args.cohort,
        "mtype-distributions_{}_{}_{}_samps-{}.png".format(
            args.gene, args.mut_levels.replace('__', '-'),
            args.classif, args.samp_cutoff
            )
        ),
        dpi=300, bbox_inches='tight')

    plt.close()


def plot_gene_isolation(mtypes, infer_dict, cdata, args):
    fig, axarr = plt.subplots(figsize=(0.1 + 3 * len(mtypes), 9.2),
                              nrows=3, ncols=len(mtypes), sharex=True)

    all_mtype = MuType(cdata.train_mut.allkey())
    for j, cur_mtype in enumerate(mtypes):
        cur_mcombs = {'All': cur_mtype,
                      'Ex': ExMcomb(cdata.train_mut, cur_mtype)}

        mtype_str = str(cur_mtype).split(':')[-1]
        cur_stats = {lbl: np.array(cdata.train_pheno(mtp))
                     for lbl, mtp in cur_mcombs.items()}

        axarr[0, j].text(0.5, 1.03,
                         "{} mutations\n({} affected samples)".format(
                             mtype_str, np.sum(cur_stats['All'])),
                         size=13, ha='center', va='bottom',
                         transform=axarr[0, j].transAxes)

        for xval in [0, 2]:
            axarr[2, j].text(xval, -0.05, "all other\nsamps",
                             ha='center', va='top', size=9)

        for xval in [1, 3]:
            axarr[2, j].text(xval, -0.05,
                             "only {}\nWT samps".format(args.gene),
                             ha='center', va='top', size=9)

        axarr[2, j].text(0.5, -0.18,
                         "all samps w/\n{} muts".format(mtype_str),
                         ha='center', va='top', size=9)
        axarr[2, j].text(2.5, -0.18,
                         "samps w/ only\n{} muts".format(mtype_str),
                         ha='center', va='top', size=9)

        infer_vals = pd.DataFrame({
            "{}_{}".format(lbl, smps): vals.loc[cur_mcombs[lbl]]
            for (lbl, stat), (smps, vals) in product(cur_stats.items(),
                                                     infer_dict.items())
            })

        infer_vals = ((infer_vals - infer_vals.min())
                      / (infer_vals.max() - infer_vals.min()))
        for lbl, stat in cur_stats.items():
            infer_vals["cStat_{}".format(lbl)] = stat

        iso_mtypes = [(lbl, (all_mtype & mtype) - cur_mtype)
                      if mtype.is_supertype(cur_mtype) else (lbl, mtype)
                      for lbl, mtype in variant_mtypes]

        for i, (lbl, other_mtype) in enumerate(iso_mtypes):
            val_df = infer_vals.copy()
            val_df['oStat'] = np.array(cdata.train_pheno(other_mtype))

            if j == 0:
                axarr[i, j].text(-0.04, 0.5,
                                 "{}\n({} samples)".format(
                                     lbl, np.sum(val_df['oStat'])),
                                 ha='right', va='center', size=11,
                                 transform=axarr[i, j].transAxes)

            val_df = pd.melt(val_df, id_vars=['cStat_All', 'cStat_Ex',
                                              'oStat'],
                             var_name='Samps', value_name='Val')

            val_df['cStat'] = np.where(
                val_df.Samps.str.split('_').apply(itemgetter(0)) == 'All',
                val_df.cStat_All, val_df.cStat_Ex
                )

            sns.violinplot(data=val_df[~val_df.cStat], x='Samps', y='Val',
                           hue='oStat', palette=[mut_clrs['Wild-Type']],
                           linewidth=0, bw=0.11, split=True, cut=0,
                           order=['All_All', 'All_Iso', 'Ex_All', 'Ex_Iso'],
                           ax=axarr[i, j])

            sns.violinplot(data=val_df[val_df.cStat], x='Samps', y='Val',
                           hue='oStat', palette=[mut_clrs['Mutant']],
                           linewidth=0, bw=0.11, split=True, cut=0,
                           order=['All_All', 'All_Iso', 'Ex_All', 'Ex_Iso'],
                           ax=axarr[i, j])

            axarr[i, j].xaxis.label.set_visible(False)
            axarr[i, j].yaxis.label.set_visible(False)
            axarr[i, j].set_xticklabels([])
            axarr[i, j].set_yticklabels([])
            axarr[i, j].set_ylim(-0.02, 1.02)

            for xval in range(4):
                axarr[i, j].axvline(x=xval, ymin=-1, ymax=2,
                                    color='black', linewidth=1.3, alpha=0.61)

            axarr[i, j].get_legend().remove()
            for art in axarr[i, j].get_children():
                if isinstance(art, PolyCollection):
                    art.set_alpha(0.41)

        axarr[2, j].legend(
            [Patch(color=mut_clrs['Mutant'], alpha=0.36),
             Patch(color=mut_clrs['Wild-Type'], alpha=0.36)],
            ["{} Mutants".format(mtype_str),
             "{} Wild-Types".format(mtype_str)],
            fontsize=11, ncol=1, loc=9, bbox_to_anchor=(0.5, -0.26)
            ).get_frame().set_linewidth(0.0)

    axarr[2, 0].text(-1, -0.05, "Negative\nClassification Set",
                     ha='right', va='top', size=10)
    axarr[2, 0].text(-1, -0.17, "Positive\nClassification Set",
                     ha='right', va='top', size=10)

    plt.tight_layout()
    plt.savefig(os.path.join(
        plot_dir, args.cohort,
        "gene-isolation_{}_{}_{}_samps-{}.png".format(
            args.gene, args.mut_levels.replace('__', '-'),
            args.classif, args.samp_cutoff
            )
        ),
        dpi=300, bbox_inches='tight')

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        "Plot the ordering of a gene's subtypes in a given cohort based on "
        "how their isolated expression signatures classify one another."
        )

    parser.add_argument('cohort', help='a TCGA cohort')
    parser.add_argument('gene', help='a mutated gene')
    parser.add_argument('classif', help='a mutation classifier')
    parser.add_argument('mut_levels', default='Protein',
                        help='a set of mutation annotation levels')
    parser.add_argument('--samp_cutoff', default=20)

    # parse command line arguments, create directory where plots will be saved
    args = parser.parse_args()
    os.makedirs(os.path.join(plot_dir, args.cohort), exist_ok=True)

    cdata = load_cohort_data(base_dir,
                             args.cohort, args.gene, args.mut_levels)
    use_lvls = args.mut_levels.split('__')

    infer_dict = load_infer_output(os.path.join(
        base_dir, 'output', args.cohort, args.gene, args.classif,
        'samps_{}'.format(args.samp_cutoff), args.mut_levels
        ))

    assert set(infer_dict['Iso'].index) == set(infer_dict['All'].index)
    infer_dict = {smps: vals.applymap(np.mean)
                  for smps, vals in infer_dict.items()}

    base_mtypes = sorted(mtype for mtype in infer_dict['All'].index
                         if (isinstance(mtype, MuType)
                             and use_lvls[-1] in mtype.get_levels()
                             and (ExMcomb(cdata.train_mut, mtype)
                                  in infer_dict['Iso'].index)))

    plot_mtype_distributions(base_mtypes, infer_dict.copy(), cdata, args)
    plot_gene_isolation(base_mtypes, infer_dict.copy(), cdata, args)


if __name__ == '__main__':
    main()

