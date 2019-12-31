
import os
import sys

base_dir = os.path.join(os.environ['DATADIR'], 'HetMan', 'subvariant_test')
sys.path.extend([os.path.join(os.path.dirname(__file__), '..', '..', '..')])
plot_dir = os.path.join(base_dir, 'plots', 'mutations')

from HetMan.experiments.subvariant_test import (
    domain_dir, pnt_mtype, copy_mtype)
from HetMan.experiments.subvariant_test.merge_test import merge_cohort_data
from HetMan.experiments.subvariant_tour.utils import RandomType
from HetMan.experiments.subvariant_test.utils import get_fancy_label
from HetMan.experiments.subvariant_test.plot_copy import select_mtype
from HetMan.experiments.subvariant_tour.plot_aucs import choose_gene_colour

from HetMan.experiments.subvariant_infer import variant_clrs
from dryadic.features.data.domains import get_protein_domains
from dryadic.features.mutations import MuType, MuTree

import argparse
from pathlib import Path
import bz2
import dill as pickle

import numpy as np
import pandas as pd

from operator import itemgetter
import re
import random

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.patches import Rectangle as Rect
from matplotlib.collections import PatchCollection

# make plots cleaner by turning off outer box, make background all white
plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'
plt.rcParams['axes.edgecolor']='white'


def clean_level(lvl):
    if 'Domain_' in lvl:
        use_lvl = "{} Domain".format(lvl.split('Domain_')[1])

    elif '_base' in lvl:
        use_lvl = "{}(base)".format(lvl.split('_base')[0])

    elif lvl == 'Location':
        use_lvl = "Amino Acid"

    elif lvl == 'Protein':
        use_lvl = "AA Substitution"

    else:
        use_lvl = lvl

    return use_lvl


def clean_label(lbl, lvl):
    if lvl == 'Exon':
        use_lbl = lbl.split('/')[0]

    elif lvl == 'Protein':
        use_lbl = re.sub("[0-9]+", "->", lbl, count=1)[2:]

    elif 'Form' in lvl:
        use_lbl = lbl.replace('_Mutation', '').replace('_', '')

    elif 'Domain_' in lvl:
        use_lbl = lbl.replace('None', 'no overlapping domain')

    else:
        use_lbl = lbl

    return use_lbl


def plot_lollipop(cdata_dict, domain_dict, args):
    fig, main_ax = plt.subplots(figsize=(11, 4))

    base_cdata = tuple(cdata_dict.values())[0]
    gn_annot = base_cdata.gene_annot[args.gene]

    base_lvls = 'Gene', 'Scale', 'Form_base', 'Location', 'Protein',
    domn_lvls = [('Gene', 'Scale', 'Transcript',
                  '_'.join(['Domain', domn_nm]), 'Location')
                 for domn_nm in domain_dict]

    for lvls in [base_lvls] + domn_lvls:
        if lvls not in base_cdata.mtrees:
            base_cdata.add_mut_lvls(lvls)

    loc_mtree = base_cdata.mtrees[base_lvls][args.gene]['Point']
    #TODO: double-check mutations in cohorts such as METABRIC that are not
    # numeric but maybe should be (i.e. in CDH1, MAP3K1)
    loc_dict = {form: sorted([(int(loc), len(loc_muts)) if loc.isnumeric()
                              else (-1, len(loc_muts))
                              for loc, loc_muts in form_muts],
                             key=itemgetter(0))
                for form, form_muts in loc_mtree}

    # calculate the minimum and maximum amino acid positions for the
    # mutations to be plotted, as well as the most samples at any hotspot
    min_pos = min(pos for loc_counts in loc_dict.values()
                  for pos, _ in loc_counts if pos >= 0)
    max_pos = max(pos for loc_counts in loc_dict.values()
                  for pos, _ in loc_counts)
    max_count = max(count for loc_counts in loc_dict.values()
                    for _, count in loc_counts)

    lgnd_ptchs = []
    for form, form_muts in loc_mtree:
        form_clr = choose_gene_colour(form, clr_seed=2783,
                                      clr_lum=0.37, clr_sat=0.91)

        lgnd_ptchs += [Patch(color=form_clr, alpha=0.53,
                             label=clean_label(form, 'Form_base'))]
        mrks, stms, basl = main_ax.stem(*zip(*loc_dict[form]),
                                        use_line_collection=True)

        plt.setp(mrks, markersize=7, markeredgecolor='black',
                 markerfacecolor=form_clr, zorder=5)
        plt.setp(stms, linewidth=0.8, color='black', zorder=1)
        plt.setp(basl, linewidth=1.1, color='black', zorder=2)

        for loc, loc_muts in form_muts:
            if len(loc_muts) >= 10:
                mut_lbls = sorted(lbl for lbl, _ in loc_muts)

                root_indx = re.match('p.[A-Z][0-9]+', mut_lbls[0]).span()[1]
                lbl_root = mut_lbls[0][2:root_indx]

                if max(len(lbl) - len(lbl_root) for lbl in mut_lbls) > 4:
                    loc_lbl = '\n'.join(
                        [mut_lbls[0][2:]] + [''.join([' ' * len(lbl_root) * 2,
                                                      lbl.split(lbl_root)[1]])
                                             for lbl in mut_lbls[1:]]
                        )

                else:
                    loc_lbl = "/".join(
                        [mut_lbls[0][2:]] + [lbl.split(lbl_root)[1]
                                             for lbl in mut_lbls[1:]]
                        )

                main_ax.text(int(loc) + (max_pos - min_pos) / 115,
                             len(loc_muts) + max_count / 151,
                             loc_lbl, size=8, ha='left', va='center')

    prot_patches = []
    for i, lvls in enumerate(domn_lvls):
        tx_mtree = base_cdata.mtrees[lvls][args.gene]['Point']
        tx_id = tuple(tx_mtree)[0][0]
        tx_annot = gn_annot['Transcripts'][tx_id]

        domn_nm = lvls[3].split('_')[1]
        domn_df = domain_dict[domn_nm]
        gene_domns = domn_df[(domn_df.Gene == gn_annot['Ens'])
                             & (domn_df.Transcript == tx_id)]

        for domn_id, domn_start, domn_end in zip(gene_domns.DomainID,
                                                 gene_domns.DomainStart,
                                                 gene_domns.DomainEnd):
            prot_patches.append(Rect(
                (domn_start, -max_count * (0.3 + i * 0.13)),
                domn_end - domn_start, max_count * 0.11
                ))

            main_ax.text((domn_start + domn_end) / 2,
                         -max_count * (0.25 + i * 0.13),
                         domn_id, size=9, ha='center', va='center')

    main_ax.add_collection(PatchCollection(
        prot_patches, alpha=0.4, linewidth=0, color='#D99100'))

    exn_patches = []
    exn_pos = 1
    for i, exn_annot in enumerate(tx_annot['Exons']):
        exn_len = exn_annot['End'] - exn_annot['Start'] + 1

        if 'UTRs' in tx_annot:
            for utr_annot in tx_annot['UTRs']:
                if (exn_annot['Start'] <= utr_annot['Start']
                        <= exn_annot['End'] <= utr_annot['End']):
                    exn_len -= exn_annot['End'] - utr_annot['Start'] + 1

                elif (exn_annot['Start'] <= utr_annot['Start']
                        <= utr_annot['End'] <= exn_annot['End']):
                    exn_len -= utr_annot['End'] - utr_annot['Start'] + 1

        if exn_len > 0 and exn_pos <= max_pos:
            exn_len //= 3

            if i == (len(tx_annot['Exons']) - 1):
                if (exn_pos + exn_len) > max_pos:
                    exn_len = max_pos - exn_pos + 10

            if (exn_pos + exn_len) >= min_pos:
                exn_patches.append(Rect((exn_pos, max_count * -0.15),
                                        exn_len, max_count * 0.11,
                                        color='green'))

                main_ax.text(max(exn_pos + exn_len / 2, min_pos + 5),
                             max_count * -0.1, exn_annot['number'],
                             size=min(11, (531 * exn_len / max_pos) ** 0.6),
                             ha='center', va='center')

            exn_pos += exn_len

    for i, domn_nm in enumerate(domain_dict):
        main_ax.text(min_pos - exn_pos / 29, -max_count * (0.23 + i * 0.15),
                     "{}\nDomains".format(domn_nm), size=7,
                     ha='right', va='top', linespacing=0.65, rotation=37)

    # add the patches describing the boundaries of each exon and annotate them
    main_ax.add_collection(PatchCollection(
        exn_patches, alpha=0.4, linewidth=1.4, color='#002C91'))
    main_ax.text(min_pos - exn_pos / 29, max_count * -0.08,
                 "{}\nExons".format(tx_annot['transcript_name']), size=7,
                 ha='right', va='top', linespacing=0.65, rotation=37)

    if '_' in args.cohort:
        coh_lbl = "{}({})".format(*args.cohort.split('_'))
    else:
        coh_lbl = str(args.cohort)

    main_ax.text(
        0.03, 0.97,
        "{} {}-mutated samples\n{:.1%} of {} affected".format(
            len(loc_mtree), args.gene,
            len(loc_mtree) / len(base_cdata.get_samples()), coh_lbl,
            ),
        size=13, ha='left', va='top', transform=main_ax.transAxes
        )

    # add the legend for the colour used for each form of mutation
    plt_lgnd = main_ax.legend(handles=lgnd_ptchs, frameon=False, fontsize=11,
                              ncol=3, loc=1, handletextpad=0.7,
                              bbox_to_anchor=(0.98, 1.02))
    main_ax.add_artist(plt_lgnd)

    main_ax.grid(linewidth=0.31)
    main_ax.set_xlabel("Amino Acid Position", size=17, weight='semibold')
    main_ax.set_ylabel("       # of Mutated Samples",
                       size=17, weight='semibold')

    main_ax.set_xlim(min_pos - exn_pos / 29, exn_pos * 1.02)
    main_ax.set_ylim(-max_count * (0.03 + len(domain_dict) * 0.23),
                     max_count * 24 / 17)
    main_ax.set_yticks([tck for tck in main_ax.get_yticks() if tck >= 0])

    # save the plot to file
    fig.savefig(os.path.join(plot_dir, args.cohort,
                             "{}_lollipop.svg".format(args.gene)),
                bbox_inches='tight', format='svg')

    plt.close()


def sort_levels(lbls, lvl):
    if lvl == 'Exon' or lvl == 'Location':
        sort_indx = sorted(range(len(lbls)),
                           key=lambda k: (int(lbls[k].split('/')[0])
                                          if lbls[k].isnumeric() else 0))

    elif 'Domain_' in lvl and 'none' in lbls:
        sort_indx = [lbls.index('none')]

        sort_indx += (sorted(range(sort_indx[0]), key=lambda k: lbls[k])
                      + sorted(range(sort_indx[0] + 1, len(lbls)),
                               key=lambda k: lbls[k]))

    else:
        sort_indx = range(len(lbls))

    return [lbls[i] for i in sort_indx]


def recurse_labels(ax, mtree, xlims, ymax, all_size,
                   cur_j=0, clr_mtype=False, add_lbls=True, mut_clr=None):
    all_wdth = len(MuType(mtree.allkey()).subkeys())
    cur_x = xlims[0]
    muts_dict = dict(mtree)

    if mut_clr is None:
        mut_clr = variant_clrs['Point']

    for lbl in sort_levels(list(muts_dict.keys()), mtree.mut_level):
        muts = muts_dict[lbl]

        if isinstance(muts, MuTree):
            lf_wdth = len(MuType(muts.allkey()).subkeys())
        else:
            lf_wdth = 1

        lbl_prop = (lf_wdth / all_wdth) * (xlims[1] - xlims[0])
        ax.plot([cur_x + lbl_prop / 2, (xlims[0] + xlims[1]) / 2],
                [ymax - 0.18 - cur_j, ymax + 0.19 - cur_j],
                c='black', linewidth=0.9, solid_capstyle='round')

        if add_lbls:
            mut_lbl = clean_label(lbl, mtree.mut_level)
            if (lbl_prop / all_size) > 1/41 and len(tuple(mtree)) > 1:
                if len(muts) == 1:
                    mut_lbl = "{}\n(1 sample)".format(mut_lbl)
                else:
                    mut_lbl = "{}\n({} samps)".format(mut_lbl, len(muts))

            if (lbl_prop / all_size) <= 1/19:
                use_rot = 90
            else:
                use_rot = 0

            ax.text(cur_x + lbl_prop / 2, ymax - 0.5 - cur_j, mut_lbl,
                    size=14 - 2.9 * cur_j, ha='center', va='center',
                    rotation=use_rot)

        # colour this branch and all subtypes
        if clr_mtype is None or clr_mtype is False:
            use_clr = mut_clr
            eg_clr = mut_clr
            sub_mtype = clr_mtype

        # do not colour this branch or any of its subtypes
        elif clr_mtype.is_empty():
            use_clr = variant_clrs['WT']
            eg_clr = variant_clrs['WT']
            sub_mtype = clr_mtype

        # otherwise, check to see which subtypes need to be coloured
        else:
            sub_dict = dict(clr_mtype.subtype_list())

            if lbl not in sub_dict:
                use_clr = variant_clrs['WT']
                eg_clr = variant_clrs['WT']
                sub_mtype = MuType({})

            elif (isinstance(mtree[lbl], dict) or sub_dict[lbl] is None
                    or (sub_dict[lbl] == MuType(mtree[lbl].allkey()))):
                use_clr = mut_clr
                eg_clr = mut_clr
                sub_mtype = None

            else:
                use_clr = 'none'
                eg_clr = mut_clr
                sub_mtype = sub_dict[lbl]

        if clr_mtype is False:
            sub_lbls = True
        else:
            sub_lbls = False

        ax.add_patch(Rect((cur_x + lbl_prop * 0.12, ymax - 0.8 - cur_j),
                          lbl_prop * 0.76, 0.6, facecolor=use_clr,
                          edgecolor=eg_clr, alpha=0.41, linewidth=1.3))

        if isinstance(muts, MuTree):
            ax = recurse_labels(
                ax, muts, (cur_x + lbl_prop * 0.06, cur_x + lbl_prop * 0.94),
                ymax, all_size, cur_j + 1, sub_mtype, sub_lbls, mut_clr
                )

        cur_x += lbl_prop

    return ax


def plot_tree_classif(pred_dict, phn_dict, auc_dict, use_lvls,
                      cdata_dict, args):
    base_cdata = tuple(cdata_dict.values())[0]
    base_lvls = 'Exon__Location__Protein'

    lvls_k = tuple(use_lvls.split('__'))
    base_cdata.add_mut_lvls(('Gene', ) + lvls_k)
    use_mtree = base_cdata.mtrees[('Gene', ) + lvls_k][args.gene]

    use_mtypes = {(src, clf): [MuType({('Gene', args.gene): pnt_mtype})]
                  for src, lvls, clf in pred_dict if lvls == use_lvls}

    use_criter = {
        (src, clf): [(np.mean(phn_dict[src, base_lvls, clf][mtypes[0]]),
                      auc_dict[src, base_lvls, clf].loc[mtypes[0]])]
        for (src, clf), mtypes in use_mtypes.items()
        }

    for src, clf in use_mtypes:
        cur_mtypes = {mtype.subtype_list()[0][1]
                      for mtype, phn in phn_dict[src, use_lvls, clf].items()
                      if (not isinstance(mtype, RandomType)
                          and mtype.subtype_list()[0][1] != pnt_mtype
                          and (mtype.subtype_list()[0][1]
                               & copy_mtype).is_empty())}

        if len(cur_mtypes) >= 5:
            use_mtypes[src, clf] += [
                MuType({('Gene', args.gene): mtype})
                for mtype in random.sample(sorted(cur_mtypes), k=5)
                ]

            use_criter[src, clf] += [
                (np.mean(phn_dict[src, use_lvls, clf][mtype]),
                 auc_dict[src, use_lvls, clf].loc[mtype])
                for mtype in use_mtypes[src, clf][1:]
                ]

    use_src, use_clf = sorted(
        use_criter.items(),
        key=lambda x: np.prod(np.var(np.array(x[1]), axis=0))
        )[-1][0]
    plt_mtypes = use_mtypes[use_src, use_clf]

    fig = plt.figure(figsize=(1.1 + 3.1 * len(plt_mtypes), 12))
    gs = gridspec.GridSpec(nrows=3, ncols=len(plt_mtypes) + 1,
                           width_ratios=[1] + [3] * len(plt_mtypes),
                           height_ratios=[2, 1, 4])

    lbl_ax = fig.add_subplot(gs[:, 0])
    lbl_ax.axis('off')

    tree_ax = fig.add_subplot(gs[0, 1:])
    tree_ax.axis('off')
    leaf_count = len(MuType(use_mtree.allkey()).subkeys())

    tree_ax.add_patch(Rect((leaf_count * 0.03, len(lvls_k) + 0.17),
                           leaf_count * 0.23, 0.93,
                           facecolor=variant_clrs['WT'], alpha=0.41,
                           clip_on=False, linewidth=0))

    tree_ax.text(
        leaf_count * 0.15, len(lvls_k) + 0.59,
        "Wild-Type for\n{} Point Mutations\n({} samples)".format(
            args.gene, len(set(base_cdata.get_samples())
                           - set(use_mtree.get_samples()))
            ),
        size=19, ha='center', va='center',
        )

    tree_ax.add_patch(Rect((leaf_count * 0.31, len(lvls_k) + 0.23),
                           leaf_count * 0.43, 0.79,
                           facecolor=variant_clrs['Point'], alpha=0.41,
                           clip_on=False, linewidth=0))

    tree_ax.text(leaf_count / 2, len(lvls_k) + 0.61,
                 "All {} Point Mutations\n({} samples)".format(
                     args.gene, len(use_mtree)),
                 size=19, ha='center', va='center')

    tree_ax = recurse_labels(tree_ax, use_mtree, (0, leaf_count), len(lvls_k),
                             leaf_count, clr_mtype=False, add_lbls=True)
    tree_ax.set_xlim(0, leaf_count * 1.03)
    tree_ax.set_ylim(0, len(lvls_k) + 0.6)

    for i, lvl in enumerate(lvls_k):
        lbl_ax.text(1.31, 0.88 - i / 10.11, clean_level(lvl),
                    size=19, ha='right', va='center')

    for i, plt_mtype in enumerate(plt_mtypes):
        mtype_ax = fig.add_subplot(gs[1, i + 1])
        mtype_ax.axis('off')

        if plt_mtype == MuType({('Gene', args.gene): pnt_mtype}):
            tree_mtype = None
            top_fc = variant_clrs['Point']
            top_ec = 'none'

        else:
            tree_mtype = plt_mtype.subtype_list()[0][1].subtype_list()[0][1]
            top_fc = 'none'
            top_ec = variant_clrs['Point']

        mtype_ax.add_patch(Rect((leaf_count * 0.19, len(lvls_k) - 0.19),
                                leaf_count * 0.67, 0.31, clip_on=False,
                                facecolor=top_fc, edgecolor=top_ec,
                                alpha=0.41, linewidth=2.7))

        mtype_ax = recurse_labels(mtype_ax, use_mtree, (0, leaf_count),
                                  len(lvls_k) - 0.4, leaf_count,
                                  clr_mtype=tree_mtype, add_lbls=False)

        mtype_ax.set_xlim(0, leaf_count * 1.03)
        mtype_ax.set_ylim(0, len(lvls_k) + 0.6)

        if i == 0:
            pred_vals = pred_dict[use_src, base_lvls, use_clf].loc[plt_mtype]
            use_phn = phn_dict[src, base_lvls, clf][plt_mtype]

        else:
            pred_vals = pred_dict[use_src, use_lvls, use_clf].loc[plt_mtype]
            use_phn = phn_dict[src, use_lvls, clf][plt_mtype]

        pred_vals = pred_vals.apply(np.mean)
        use_auc = use_criter[src, clf][i][1]
        viol_ax = fig.add_subplot(gs[2, i + 1])

        sns.violinplot(x=pred_vals[~use_phn].values,
                       ax=viol_ax, palette=[variant_clrs['WT']],
                       orient='v', linewidth=0, cut=0, width=0.67)
        sns.violinplot(x=pred_vals[use_phn].values,
                       ax=viol_ax, palette=[variant_clrs['Point']],
                       orient='v', linewidth=0, cut=0, width=0.67)

        viol_ax.text(0.5, 1.01,
                     '\n'.join(get_fancy_label(plt_mtype).split('\n')[1:]),
                     size=12, ha='center', va='top',
                     transform=viol_ax.transAxes)

        viol_ax.text(1.07, 0.83,
                     '\n'.join([str(np.sum(use_phn)), "mutated", "samples"]),
                     size=13, ha='right', va='top', c=variant_clrs['Point'],
                     weight='semibold', transform=viol_ax.transAxes)

        viol_ax.text(0.5, -0.05, "AUC: {:.3f}".format(use_auc),
                     size=21, ha='center', va='bottom',
                     transform=viol_ax.transAxes)

        viol_ax.get_children()[0].set_alpha(0.41)
        viol_ax.get_children()[2].set_alpha(0.41)
        viol_ax.set_yticklabels([])

        viol_ylims = viol_ax.get_ylim()
        ylim_gap = (viol_ylims[1] - viol_ylims[0]) / 13
        viol_ax.set_ylim([viol_ylims[0], viol_ylims[1] + ylim_gap])

    # save the plot to file
    fig.tight_layout(w_pad=1.9, h_pad=1.5)
    fig.savefig(
        os.path.join(plot_dir, args.cohort,
                     "{}_tree-classif__{}.svg".format(args.gene, use_lvls)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        "Plots the incidence and structure of the variants of a gene "
        "present in the samples of a particular cohort."
        )

    parser.add_argument('cohort', help='a TCGA cohort')
    parser.add_argument('gene', help='a mutated gene')
    args = parser.parse_args()

    out_datas = [
        out_file.parts[-2:] for out_file in Path(base_dir).glob(
            "*__{}__samps-*/trnsf-vals__*__*.p.gz".format(args.cohort))
        ]

    out_list = pd.DataFrame([
        {'Source': out_data[0].split("__{}".format(args.cohort))[0],
         'Samps': int(out_data[0].split("__samps-")[1]),
         'Levels': '__'.join(out_data[1].split(
             "trnsf-vals__")[1].split('__')[:-1]),
         'Classif': out_data[1].split('__')[-1].split(".p.gz")[0]}
        for out_data in out_datas
        ])

    if out_list.shape[0] == 0:
        raise ValueError("No experiment output found for tumour "
                         "cohort `{}` !".format(args.cohort))

    out_use = out_list.groupby(['Source', 'Classif']).filter(
        lambda outs: ('Exon__Location__Protein' in set(outs.Levels)
                      and outs.Levels.str.match('Domain_').any())
        ).groupby(['Source', 'Levels', 'Classif'])['Samps'].min()

    cdata_dict = {
        (src, clf): merge_cohort_data(
            os.path.join(base_dir,
                         "{}__{}__samps-{}".format(
                             src, args.cohort,
                             outs.loc[(slice(None), 'Exon__Location__Protein',
                                       slice(None))][0]
                            )
                        ),
            use_seed=8713
            )
        for (src, clf), outs in out_use.groupby(['Source', 'Classif'])
        }

    domn_dict = {
        domn: get_protein_domains(domain_dir, domn)
        for domn in {lvls.split('Domain_')[1].split('__')[0]
                     for lvls in set(out_use.index.get_level_values('Levels'))
                     if 'Domain_' in lvls}
        }

    pred_dict = dict()
    phn_dict = dict()
    auc_dict = dict()

    for (src, lvls, clf), ctf in out_use.iteritems():
        out_tag = "{}__{}__samps-{}".format(src, args.cohort, ctf)

        pred_fl = os.path.join(base_dir, out_tag,
                               "out-pred__{}__{}.p.gz".format(lvls, clf))
        phn_fl = os.path.join(base_dir, out_tag,
                              "out-pheno__{}__{}.p.gz".format(lvls, clf))
        auc_fl = os.path.join(base_dir, out_tag,
                              "out-aucs__{}__{}.p.gz".format(lvls, clf))

        with bz2.BZ2File(pred_fl, 'r') as f:
            pred_data = pickle.load(f)

            pred_dict[src, lvls, clf] = pred_data.loc[[
                mtype for mtype in pred_data.index
                if select_mtype(mtype, args.gene)
                ]]

        with bz2.BZ2File(phn_fl, 'r') as f:
            phns = pickle.load(f)

            phn_dict[src, lvls, clf] = {
                mtype: phn for mtype, phn in phns.items()
                if select_mtype(mtype, args.gene)
                }

        with bz2.BZ2File(auc_fl, 'r') as f:
            auc_data = pickle.load(f)['mean']

            auc_dict[src, lvls, clf] = auc_data[[
                mtype for mtype in auc_data.index
                if select_mtype(mtype, args.gene)
                ]]

    if not any(len(phn_vals) > 0 for phn_vals in phn_dict.values()):
        raise ValueError("No experiment output found for "
                         "gene `{}` !".format(args.gene))

    os.makedirs(os.path.join(plot_dir, args.cohort), exist_ok=True)
    plot_lollipop(cdata_dict, domn_dict, args)
    for lvls in set(out_use.index.get_level_values('Levels')):
        plot_tree_classif(pred_dict, phn_dict, auc_dict, lvls,
                          cdata_dict, args)


if __name__ == '__main__':
    main()

