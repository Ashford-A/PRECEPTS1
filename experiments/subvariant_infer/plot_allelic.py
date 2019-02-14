
import os
import sys

if 'DATADIR' in os.environ:
    base_dir = os.path.join(os.environ['DATADIR'],
                            'HetMan', 'subvariant_infer')
else:
    base_dir = os.path.dirname(__file__)

plot_dir = os.path.join(base_dir, 'plots', 'allelic')
sys.path.extend([os.path.join(os.path.dirname(__file__), '../../..')])

from HetMan.experiments.subvariant_infer.setup_infer import ExMcomb
from HetMan.experiments.subvariant_infer.fit_infer import load_cohort_data
from HetMan.experiments.subvariant_infer.utils import load_infer_output
from HetMan.experiments.subvariant_infer import (
    variant_mtypes, variant_clrs, mcomb_clrs)

import argparse
import numpy as np
from scipy.stats import linregress

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Patch
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D


def plot_point_isolation(alle_vals, infer_dict, cdata, args):
    fig, axarr = plt.subplots(figsize=(12, 9), nrows=2, ncols=2, sharex=True)

    pnt_mtypes = (
        ('All', dict(variant_mtypes)['Point']),
        ('Ex', ExMcomb(cdata.train_mut, dict(variant_mtypes)['Point']))
        )
    var_stats = {lbl: np.array(cdata.train_pheno(mtype))
                 for lbl, mtype in variant_mtypes}

    wt_stats = {lbl: ~stat & ~var_stats['Gain'] & ~var_stats['Loss']
                if lbl == 'Point' else stat & ~var_stats['Point']
                for lbl, stat in var_stats.items()}
    mut_stats = {lbl: stat & ~var_stats['Gain'] & ~var_stats['Loss']
                 if lbl == 'Point' else stat & var_stats['Point']
                 for lbl, stat in var_stats.items()}

    use_alle = np.array([np.max(alle_df.alt_count
                                / (alle_df.alt_count + alle_df.ref_count))
                         if not alle_df.empty else 0
                         for alle_df in alle_vals])

    lgnd_ptchs = np.array([
        Patch(color=variant_clrs['WT'], alpha=0.51,
              label="{} point WT\nwo/ alteration".format(args.gene)),
        Patch(color=variant_clrs['Loss'], alpha=0.51,
              label="{} point WT\nw/ loss alteration".format(args.gene)),
        Patch(color=variant_clrs['Gain'], alpha=0.51,
              label="{} point WT\nw/ gain alteration".format(args.gene)),
        Line2D([0], [0], marker='o', color=variant_clrs['Point'],
               linewidth=0, markersize=12, alpha=0.71,
               label="{} point mut\nwo/ alteration".format(args.gene)),
        Line2D([0], [0], marker='D', color=mcomb_clrs['Point+Loss'],
               linewidth=0, markersize=12, alpha=0.71,
               label="{} point mut\nw/ loss alteration".format(args.gene)),
        Line2D([0], [0], marker='D', color=mcomb_clrs['Point+Gain'],
               linewidth=0, markersize=12, alpha=0.71,
               label="{} point mut\nw/ gain alteration".format(args.gene))
        ])

    pos_indx = {'All': [3, 4, 5], 'Ex': [3]}
    neg_indx = {'All': [0, 1, 2], 'Iso': [0]}

    for i, (lbl, mtype) in enumerate(pnt_mtypes):
        for j, smps in enumerate(['All', 'Iso']):
            use_vals = infer_dict[smps].loc[mtype]

            vals_min, vals_max = use_vals.quantile(q=[0, 1])
            vals_rng = (vals_max - vals_min) / 19
            axarr[i, j].set_xlim(-0.09, 1.01)
            axarr[i, j].set_ylim(vals_min - 2 * vals_rng, vals_max + vals_rng)

            for mut_lbl, mut_stat in mut_stats.items():
                use_clr = variant_clrs['Point']
                use_mrk = 'o'

                if mut_lbl != 'Point':
                    use_clr = mcomb_clrs['Point+{}'.format(mut_lbl)]
                    use_mrk = 'D'

                axarr[i, j].scatter(use_alle[mut_stat], use_vals[mut_stat],
                                    marker=use_mrk, c=use_clr, s=12,
                                    alpha=0.45, edgecolor='none')

                regr_slp, regr_intc, regr_rval, regr_pval, _ = linregress(
                    use_alle[mut_stat], use_vals[mut_stat])

                if mut_lbl == 'Point' or regr_pval < 0.01:
                    ln_xvals = np.array([0.1, 0.95, 0.98])
                    ln_yvals = regr_intc + regr_slp * ln_xvals
                    axarr[i, j].plot(ln_xvals[:2], ln_yvals[:2], '--',
                                     color=use_clr, linewidth=2.1, alpha=0.29)

                    axarr[i, j].text(0.98, ln_yvals[-1] * 1.03,
                                     "pval:{:#8.2g}\nR2:{:#7.3f}".format(
                                         regr_pval, regr_rval),
                                     size=11, color=use_clr, alpha=0.73,
                                     ha='right', va='bottom',
                                     transform=axarr[i, j].transData)

            axarr[i, j].set_yticklabels([])
            vio_ax = inset_axes(
                axarr[i, j], width='100%', height='100%', loc=10, borderpad=0,
                bbox_to_anchor=(-0.08, vals_min - 2 * vals_rng,
                                0.16, vals_max - vals_min + 3 * vals_rng),
                bbox_transform=axarr[i, j].transData
                )

            for wt_lbl, wt_stat in wt_stats.items():
                use_clr = variant_clrs[wt_lbl]
                if wt_lbl == 'Point':
                    use_clr = variant_clrs['WT']

                sns.violinplot(y=use_vals[wt_stat], ax=vio_ax, linewidth=0,
                               palette=[use_clr], cut=0)

            vio_ax.set_ylim(vals_min - 2 * vals_rng, vals_max + vals_rng)
            vio_min, vio_max = vio_ax.transAxes.inverted().transform(
                vio_ax.transData.transform([[0, yval] for yval in use_vals[
                    ~var_stats['Point']].quantile(q=[0, 1])])
                )[:, 1]

            vio_ax.axvline(x=0, ymin=vio_min, ymax=vio_max,
                           color='black', linewidth=0.7, alpha=0.47)
            vio_ax.axis('off')

            for art in vio_ax.get_children():
                if isinstance(art, PolyCollection):
                    art.set_alpha(0.43)

            ax_lgnds = [axarr[i, j].legend(
                handles=lgnd_ptchs[neg_indx[smps]].tolist(),
                frameon=False, fontsize=8, ncol=1, loc=8, handletextpad=0.5,
                bbox_to_anchor=(0.31, 0.05)
                )]

            ax_lgnds += [axarr[i, j].legend(
                handles=lgnd_ptchs[pos_indx[lbl]].tolist(),
                frameon=False, fontsize=8, ncol=1, loc=8, handletextpad=0.5,
                bbox_to_anchor=(0.59, 0.05)
                )]

            if len(pos_indx[lbl] + neg_indx[smps]) < len(lgnd_ptchs):
                use_hndls = [hndl for i, hndl in enumerate(lgnd_ptchs)
                             if i not in pos_indx[lbl] + neg_indx[smps]]

                ax_lgnds += [axarr[i, j].legend(
                    handles=use_hndls, frameon=False, fontsize=8, ncol=1,
                    loc=8, handletextpad=0.5, bbox_to_anchor=(0.87, 0.05)
                    )]

            for k, clf_lbl in enumerate(['Negative Class', 'Positive Class',
                                         'Excluded Samples']):
                axarr[i, j].text(0.31 + 0.28 * k, 0.05, clf_lbl,
                                 size=11, ha='center', va='top',
                                 transform=axarr[i, j].transAxes)

            for lgnd in ax_lgnds[:-1]:
                axarr[i, j].add_artist(lgnd)

    plt.text(0.5, 0.01, 'Variant Allele Frequency',
             size=18, ha='center', va='top', fontweight='semibold',
             transform=fig.transFigure)

    for j in range(2):
        axarr[1, j].set_xticklabels(['WT' if tck == 0
                                     else '{:.2f}'.format(tck)
                                     for tck in axarr[1, j].get_xticks()])

    plt.tight_layout()
    plt.savefig(os.path.join(
        plot_dir, args.cohort,
        "point-isolation_{}_{}_{}_samps-{}.svg".format(
            args.gene, args.mut_levels.replace('__', '-'),
            args.classif, args.samp_cutoff
            )
        ),
        dpi=400, bbox_inches='tight', format='svg')

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        "Plot the association between inferred subvariant classifier scores "
        "and the associated mutations' variant allele frequencies."
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
    alle_vals = [cdata.alleles[cdata.alleles.Sample == samp].drop_duplicates()
                 for samp in sorted(cdata.train_samps)]

    infer_dict = load_infer_output(os.path.join(
        base_dir, 'output', args.cohort, args.gene, args.classif,
        'samps_{}'.format(args.samp_cutoff), args.mut_levels
        ))

    assert set(infer_dict['Iso'].index) == set(infer_dict['All'].index)
    infer_dict = {smps: vals.applymap(np.mean)
                  for smps, vals in infer_dict.items()}

    plot_point_isolation(alle_vals.copy(), infer_dict.copy(), cdata, args)


if __name__ == '__main__':
    main()

