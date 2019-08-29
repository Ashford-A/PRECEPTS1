
import os
import sys

base_dir = os.path.join(os.environ['DATADIR'], 'HetMan', 'subvariant_tour')
sys.path.extend([os.path.join(os.path.dirname(__file__), '../../..')])
plot_dir = os.path.join(base_dir, 'plots', 'gene')

from HetMan.experiments.subvariant_tour import *
from HetMan.experiments.subvariant_tour import pnt_mtype
from HetMan.experiments.subvariant_tour.merge_tour import merge_cohort_data
from HetMan.experiments.subvariant_tour.utils import RandomType
from HetMan.experiments.subvariant_tour.plot_aucs import get_fancy_label

from HetMan.experiments.subvariant_infer import variant_clrs
from dryadic.features.data.domains import get_protein_domains
from dryadic.features.mutations import MuType, MuTree

import argparse
import glob as glob
from pathlib import Path
import bz2
import dill as pickle

import numpy as np
import pandas as pd

from functools import reduce
from operator import or_, itemgetter
import re
import random

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle as Rect
from matplotlib.collections import PatchCollection
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# make plots cleaner by turning off outer box, make background all white
plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'
plt.rcParams['axes.edgecolor']='white'


def plot_mutation_lollipop(cdata_dict, domain_dict, args):
    fig, main_ax = plt.subplots(figsize=(10, 4))

    base_cdata = tuple(cdata_dict.values())[0]['Loc']
    use_mtree = base_cdata.mtree[args.gene]['Point']

    loc_counts = sorted([(int(loc), len(loc_muts))
                         for exn, exn_muts in use_mtree
                         for loc, loc_muts in exn_muts if loc != '.'],
                        key=itemgetter(0))
    max_count = max(count for _, count in loc_counts)

    mrks, stms, basl = main_ax.stem(*zip(*loc_counts))
    plt.setp(mrks, markersize=5, markeredgecolor='black', zorder=5)
    plt.setp(stms, linewidth=0.8, color='black', zorder=1)
    plt.setp(basl, linewidth=1.1, color='black', zorder=2)

    for exn, exn_muts in use_mtree:
        for loc, loc_muts in exn_muts:
            if len(loc_muts) >= 10:
                mut_lbls = sorted(lbl for lbl, _ in loc_muts)
                root_indx = re.match('p.[A-Z][0-9]+', mut_lbls[0]).span()[1]
                lbl_root = mut_lbls[0][2:root_indx]

                main_ax.text(
                    int(loc) + (loc_counts[-1][0] - loc_counts[0][0]) / 173,
                    len(loc_muts) + max_count / 173,
                    "/".join([mut_lbls[0][2:]] + [lbl.split(lbl_root)[1]
                                                  for lbl in mut_lbls[1:]]),
                    size=8, ha='left', va='bottom'
                    )

    gn_annot = base_cdata.gene_annot[args.gene]
    main_tx = [
        tx_id for tx_id, tx_annot in gn_annot['Transcripts'].items()
        if tx_annot['transcript_name'] == '{}-001'.format(args.gene)
        ][0]

    prot_patches = []
    gene_doms = domain_dict['SMART'][
        (domain_dict['SMART'].Gene == gn_annot['Ens'])
        & (domain_dict['SMART'].Transcript == main_tx)
        ]

    for dom_id, dom_start, dom_end in zip(gene_doms.DomainID,
                                          gene_doms.DomainStart,
                                          gene_doms.DomainEnd):

        prot_patches.append(Rect((dom_start, max_count * -0.12),
                                 dom_end - dom_start, max_count * 0.08))
        main_ax.text((dom_start + dom_end) / 2, max_count * -0.086, dom_id,
                     size=9, ha='center', va='center')

    main_ax.add_collection(PatchCollection(
        prot_patches, alpha=0.4, linewidth=0, color='#D99100'))

    exn_patches = []
    exn_pos = 1

    for i, exn_annot in enumerate(gn_annot['Exons']):
        exn_len = exn_annot['End'] - exn_annot['Start'] + 1

        if 'UTR' in exn_annot:
            for utr_annot in exn_annot['UTR']:
                exn_len -= utr_annot['End'] - utr_annot['Start']

        if exn_len > 0 and exn_pos <= loc_counts[-1][0]:
            exn_len /= 3

            if i == (len(gn_annot['Exons']) - 1):
                if (exn_pos + exn_len) > loc_counts[-1][0]:
                    exn_len = loc_counts[-1][0] - exn_pos + 10

            if (exn_pos + exn_len) >= loc_counts[0][0]:
                exn_patches.append(Rect((exn_pos, max_count * -0.23),
                                        exn_len, max_count * 0.08,
                                        color='green'))

                main_ax.text(max(exn_pos + exn_len / 2, loc_counts[0][0] + 5),
                             max_count * -0.196,
                             "{}/{}".format(i + 1, len(gn_annot['Exons'])),
                             size=min(11, (531 * exn_len
                                           / loc_counts[-1][0]) ** 0.6),
                             ha='center', va='center')

            exn_pos += exn_len

    main_ax.add_collection(PatchCollection(
        exn_patches, alpha=0.4, linewidth=1.4, color='#002C91'))

    main_ax.text(loc_counts[0][0] - exn_pos / 25, max_count * -0.05,
                 "{}\nDomains".format('SMART'), size=7,
                 ha='right', va='top', linespacing=0.65, rotation=37)

    main_ax.text(loc_counts[0][0] - exn_pos / 25, max_count * -0.16,
                 "{}-001\nExons".format(args.gene), size=7,
                 ha='right', va='top', linespacing=0.65, rotation=37)

    main_ax.text(
        0.02, 0.79,
        "{} {}-mutated samples\n{:.1%} of {} cohort affected".format(
            len(use_mtree), args.gene,
            len(use_mtree) / len(base_cdata.get_samples()), args.cohort,
            ),
        size=11, va='bottom', transform=main_ax.transAxes
        )

    main_ax.set_xlabel("Amino Acid Position", size=15, weight='semibold')
    main_ax.set_ylabel("# of Mutated Samples", size=15, weight='semibold')
    main_ax.grid(linewidth=0.31)

    main_ax.set_xlim(loc_counts[0][0] - exn_pos / 29, exn_pos * 1.03)
    main_ax.set_ylim(max_count / -3.6, max_count * 1.07)
    main_ax.set_yticks([tck for tck in main_ax.get_yticks() if tck >= 0])

    # save the plot to file
    fig.savefig(os.path.join(
        plot_dir, "mut-lollipop_{}__{}.svg".format(args.cohort, args.gene)
        ), bbox_inches='tight', format='svg')

    plt.close()


def clean_level(lvl):
    if 'Domain_' in lvl:
        use_lvl = "{} Domain".format(lvl.split('Domain_')[1])

    elif '_base' in lvl:
        use_lvl = "{}(base)".format(lvl.split('_base')[0])

    elif lvl == 'Location':
        use_lvl = "Amino Acid"

    else:
        use_lvl = lvl

    return use_lvl


def clean_label(lbl, lvl):
    if lvl == 'Exon':
        use_lbl = lbl.split('/')[0]

    elif lvl == 'Location':
        use_lbl = "aa{}".format(lbl)

    elif lvl == 'Protein':
        use_lbl = lbl[2:].replace('del', 'd').replace('ins', 'i')

    elif 'Form' in lvl:
        use_lbl = lbl.replace('_Mutation', '').replace('_', '')

    elif 'Domain_' in lvl:
        use_lbl = lbl.replace('None', 'no overlapping domain')

    else:
        use_lbl = lbl

    return use_lbl


def recurse_labels(ax, mtree, xlims, ymax, all_size,
                   cur_j=0, clr_mtype=False, add_lbls=True):
    all_wdth = len(MuType(mtree.allkey()).subkeys())
    cur_x = xlims[0]

    for lbl, muts in mtree:
        if isinstance(muts, MuTree):
            lf_wdth = len(MuType(muts.allkey()).subkeys())
        else:
            lf_wdth = 1

        lbl_prop = (lf_wdth / all_wdth) * (xlims[1] - xlims[0])
        ax.plot([cur_x + lbl_prop / 2, (xlims[0] + xlims[1]) / 2],
                [ymax - 0.18 - cur_j, ymax + 0.19 - cur_j],
                c='0.71', linewidth=1.3, solid_capstyle='round')

        if add_lbls:
            mut_lbl = clean_label(lbl, mtree.mut_level)
            if (lbl_prop / all_size) > 1/41 and len(tuple(mtree)) > 1:
                if len(muts) == 1:
                    mut_lbl = "{}\n(1 sample)".format(mut_lbl)
                else:
                    mut_lbl = "{}\n({} samps)".format(mut_lbl, len(muts))

            if (lbl_prop / all_size) <= 1/51:
                use_rot = 90
            else:
                use_rot = 0

            ax.text(cur_x + lbl_prop / 2, ymax - 0.5 - cur_j, mut_lbl,
                    size=9 - 2.5 * cur_j, ha='center', va='center',
                    rotation=use_rot)

        if clr_mtype is False or clr_mtype == pnt_mtype:
            use_clr = variant_clrs['Point']
            sub_mtype = clr_mtype

        elif clr_mtype is None:
            use_clr = variant_clrs['Point']
            sub_mtype = None

        elif clr_mtype.is_empty():
            use_clr = variant_clrs['WT']
            sub_mtype = clr_mtype

        elif clr_mtype.subtype_list()[0][0] == lbl:
            use_clr = variant_clrs['Point']
            sub_mtype = clr_mtype.subtype_list()[0][1]

        else:
            use_clr = variant_clrs['WT']
            sub_mtype = MuType({})

        if clr_mtype is False:
            sub_lbls = True
        else:
            sub_lbls = False

        ax.add_patch(Rect((cur_x + lbl_prop * 0.12, ymax - 0.8 - cur_j),
                          lbl_prop * 0.76, 0.6, facecolor=use_clr,
                          alpha=0.41, linewidth=0))

        if isinstance(muts, MuTree):
            ax = recurse_labels(
                ax, muts, (cur_x + lbl_prop * 0.06, cur_x + lbl_prop * 0.94),
                ymax, all_size, cur_j + 1, sub_mtype, sub_lbls
                )

        cur_x += lbl_prop

    return ax


def plot_mutation_tree(cdata_dict, domain_dict, args):
    base_cdict = tuple(cdata_dict.values())[0]
    lvls_key = ['Loc'] + [lvls for lvls in set(base_cdict) - {'Loc'}]

    use_mtrees = [base_cdict[lvl_k].mtree[args.gene]['Point']
                  for lvl_k in lvls_key]
    lvls_list = [['Exon', 'Location', 'Protein']]
    lvls_list += [lvl_k.split('__') for lvl_k in sorted(lvls_key[1:])[::-1]]

    fig, axarr = plt.subplots(
        figsize=(14, 0.3 + 1.9 * len(lvls_key)), nrows=len(lvls_key), ncols=1,
        gridspec_kw=dict(
            height_ratios=[1 + len(lvls_ls) for lvls_ls in lvls_list])
        )

    for ax, use_mtree, use_lvls in zip(axarr, use_mtrees, lvls_list):
        ax.axis('off')
        leaf_count = len(MuType(use_mtree.allkey()).subkeys())

        for i, lvl in enumerate(use_lvls):
            ax.text(leaf_count / -41, len(use_lvls) - i - 0.5,
                    clean_level(lvl), size=12, ha='right', va='center')

        ax.text(leaf_count / 2, len(use_lvls) + 0.67,
                "All {} Point Mutations\n({} samples)".format(
                    args.gene, len(use_mtree)),
                size=11, ha='center', va='center')

        ax.add_patch(Rect((leaf_count * 0.29, len(use_lvls) + 0.23),
                          leaf_count * 0.42, 0.91, clip_on=False,
                          facecolor=variant_clrs['Point'], alpha=0.41,
                          linewidth=0))

        ax = recurse_labels(ax, use_mtree, (0, leaf_count), len(use_lvls),
                            leaf_count, clr_mtype=False, add_lbls=True)

        ax.set_xlim(0, leaf_count * 1.03)
        ax.set_ylim(0, len(use_lvls) + 1)

    # save the plot to file
    fig.tight_layout(h_pad=0)
    fig.savefig(os.path.join(
        plot_dir, "mut-tree_{}__{}.svg".format(args.cohort, args.gene)
        ), bbox_inches='tight', format='svg')

    plt.close()


def plot_mutation_classif(infer_dict, out_dict, cdata_dict, args):
    base_cdict = tuple(cdata_dict.values())[0]
    lvls_key = ['Loc'] * 2 + sorted(
        lvls for lvls in set(base_cdict) - {'Loc'})[::-1]

    use_mtrees = [base_cdict[lvl_k].mtree[args.gene]['Point']
                  for lvl_k in lvls_key]
    lvls_list = [['Exon', 'Location', 'Protein']] * 2
    lvls_list += [lvl_k.split('__') for lvl_k in lvls_key[2:]]

    fig = plt.figure(figsize=(12, 0.3 + 2.1 * len(lvls_key)))
    gs = gridspec.GridSpec(nrows=len(lvls_key), ncols=3,
                           width_ratios=[1, 7, 4])

    wild_ax = fig.add_subplot(gs[:, 0])
    wild_ax.axis('off')
    wild_ax.add_patch(Rect((-0.2, 0.2), 0.8, 0.6,
                           facecolor=variant_clrs['WT'],
                           clip_on=False, linewidth=0, alpha=0.41))

    wild_ax.text(
        0.2, 0.5, "Wild-Type for {} Point Mutations\n({} samples)".format(
            args.gene, len(set(base_cdict['Loc'].get_samples())
                           - set(base_cdict['Loc'].mtree[
                               args.gene]['Point'].get_samples()))
            ),
        size=21, ha='center', va='center', rotation=90
        )

    for i, (use_mtree, use_lvls, lvls_k) in enumerate(zip(
            use_mtrees, lvls_list, lvls_key)):
        tree_ax = fig.add_subplot(gs[i, 1])
        tree_ax.axis('off')

        use_outs = [
            (src, clf, auc_df.Chrm, phn_dict)
            for (src, lvls, clf), (auc_df, phn_dict) in out_dict.items()
            if lvls == '__'.join(use_lvls)
            ]

        if i == 0:
            plt_mtype = MuType({('Gene', args.gene): pnt_mtype})

        else:
            plt_mtype = MuType({
                ('Gene', args.gene): random.choice(sorted(
                    base_cdict[lvls_k].mtree[
                        args.gene]['Point'].branchtypes(min_size=25)
                    & {mtype.subtype_list()[0][1] for mtype in use_outs[0][3]
                       if not isinstance(mtype, RandomType)}
                    ))
                })

        use_src, use_clf, use_auc, use_phn = sorted(
            [(src, clf, auc_vals[plt_mtype], phn_dict[plt_mtype])
             for (src, clf, auc_vals, phn_dict) in use_outs],
            key=itemgetter(2)
            )[-1]

        leaf_count = len(MuType(use_mtree.allkey()).subkeys())
        tree_ax.add_patch(Rect((leaf_count * 0.35, len(use_lvls) + 0.23),
                               leaf_count * 0.42, 1.03, clip_on=False,
                               facecolor=variant_clrs['Point'], alpha=0.41,
                               linewidth=0))

        for j, lvl in enumerate(use_lvls):
            tree_ax.text(leaf_count / 13, len(use_lvls) - j - 0.5,
                         clean_level(lvl), size=11, ha='right', va='center')

        tree_ax = recurse_labels(
            tree_ax, use_mtree,
            (leaf_count / 11, leaf_count), len(use_lvls), leaf_count,
            clr_mtype=plt_mtype.subtype_list()[0][1], add_lbls=False
            )

        tree_ax.set_xlim(0, leaf_count * 1.03)
        tree_ax.set_ylim(0, len(use_lvls) + 1)

        infer_vals = infer_dict[
            use_src, '__'.join(use_lvls), use_clf]['Chrm'].loc[plt_mtype]
        viol_ax = fig.add_subplot(gs[i, 2])

        sns.violinplot(x=np.concatenate(infer_vals[~use_phn].values),
                       ax=viol_ax, palette=[variant_clrs['WT']],
                       linewidth=0, cut=0)
        sns.violinplot(x=np.concatenate(infer_vals[use_phn].values),
                       ax=viol_ax, palette=[variant_clrs['Point']],
                       linewidth=0, cut=0)

        viol_ax.text(-0.13, 1,
                     "{}\n({} samples)".format(get_fancy_label(plt_mtype),
                                               np.sum(use_phn)),
                     size=13, ha='left', va='top',
                     transform=viol_ax.transAxes)

        viol_ax.text(1, -0.05, "AUC: {:.3f}".format(use_auc),
                     size=14, ha='right', va='bottom',
                     transform=viol_ax.transAxes)

        viol_ax.get_children()[0].set_alpha(0.41)
        viol_ax.get_children()[2].set_alpha(0.41)
        viol_ax.set_xticklabels([])

    # save the plot to file
    fig.tight_layout(w_pad=1.1, h_pad=2.7)
    fig.savefig(os.path.join(
        plot_dir, "mut-classif_{}__{}.svg".format(args.cohort, args.gene)
        ), bbox_inches='tight', format='svg')

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        "Plots the incidence and structure of the variants of a gene "
        "present in the samples of a particular cohort."
        )

    parser.add_argument('cohort', help='a TCGA cohort')
    parser.add_argument('gene', help='a mutated gene')

    # parse command line arguments, create directory where plots will be saved
    args = parser.parse_args()
    os.makedirs(plot_dir, exist_ok=True)

    out_datas = [
        out_file.parts[-2:] for out_file in Path(base_dir).glob(
            "*__{}__samps-*/out-data__*__*.p.gz".format(args.cohort))
        ]

    out_use = pd.DataFrame([
        {'Source': out_data[0].split('__')[0],
         'Samps': int(out_data[0].split("__samps-")[1]),
         'Levels': '__'.join(out_data[1].split(
             "out-data__")[1].split('__')[:-1]),
         'Classif': out_data[1].split('__')[-1].split(".p.gz")[0]}
        for out_data in out_datas
        ]).groupby(['Source', 'Classif']).filter(
            lambda outs: ('Exon__Location__Protein' in set(outs.Levels)
                          and outs.Levels.str.match('Domain_').any())
            ).groupby(['Source', 'Levels', 'Classif'])['Samps'].min()

    cdata_dict = {
        (src, clf): {
            'Loc': merge_cohort_data(os.path.join(
                base_dir, "{}__{}__samps-{}".format(
                    src, args.cohort, outs.loc[(slice(None),
                                                'Exon__Location__Protein',
                                                slice(None))][0]
                    )
                ), 'Exon__Location__Protein', use_seed=8713),
            **{lvls: merge_cohort_data(os.path.join(
                base_dir, "{}__{}__samps-{}".format(src, args.cohort, ctf)
                ), lvls, use_seed=8713)
                for (_, lvls, _), ctf in outs.iteritems()
                if lvls != 'Exon__Location__Protein'}
            }
        for (src, clf), outs in out_use.groupby(['Source', 'Classif'])
        }

    # load protein domain data, get location of local cache for TCGA data
    domn_dict = {
        domn: get_protein_domains(domain_dir, domn)
        for domn in {lvls.split('Domain_')[1].split('__')[0]
                     for lvls in reduce(or_,
                                        [set(cdict.keys()) - {'Loc'}
                                         for cdict in cdata_dict.values()])
                     if 'Domain_' in lvls}
        }

    infer_dict = {
        (src, lvls, clf): pickle.load(bz2.BZ2File(os.path.join(
            base_dir, "{}__{}__samps-{}".format(src, args.cohort, ctf),
            "out-data__{}__{}.p.gz".format(lvls, clf)
            ), 'r'))['Infer']
        for (src, lvls, clf), ctf in out_use.iteritems()
        }
 
    out_dict = {
        (src, lvls, clf): pickle.load(bz2.BZ2File(os.path.join(
            base_dir, "{}__{}__samps-{}".format(src, args.cohort, ctf),
            "out-aucs__{}__{}.p.gz".format(lvls, clf)
            ), 'r'))
        for (src, lvls, clf), ctf in out_use.iteritems()
        }

    plot_mutation_lollipop(cdata_dict, domn_dict, args)
    plot_mutation_tree(cdata_dict, domn_dict, args)
    plot_mutation_classif(infer_dict, out_dict, cdata_dict, args)


if __name__ == '__main__':
    main()

