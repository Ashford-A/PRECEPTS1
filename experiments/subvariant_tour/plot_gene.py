
import os
import sys

base_dir = os.path.join(os.environ['DATADIR'], 'HetMan', 'subvariant_tour')
sys.path.extend([os.path.join(os.path.dirname(__file__), '../../..')])
plot_dir = os.path.join(base_dir, 'plots', 'gene')

from HetMan.experiments.subvariant_tour import *
from HetMan.experiments.subvariant_infer import variant_clrs
from HetMan.experiments.subvariant_tour.merge_tour import merge_cohort_data
from dryadic.features.data.domains import get_protein_domains

import argparse
from pathlib import Path
import glob as glob
import bz2
import dill as pickle

import pandas as pd
from functools import reduce
from operator import or_, itemgetter
import re

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

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

    base_cdict = tuple(cdata_dict.values())[0]['Loc']
    use_mtree = base_cdict.mtree[args.gene]['Point']

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

    gn_annot = base_cdict.gene_annot[args.gene]
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

    main_ax.text(loc_counts[0][0] - exn_pos / 107, max_count * -0.05,
                 "{}\nDomains".format('SMART'), size=7,
                 ha='right', va='top', linespacing=0.65, rotation=37)

    main_ax.text(loc_counts[0][0] - exn_pos / 107, max_count * -0.16,
                 "{}-001\nExons".format(args.gene), size=7,
                 ha='right', va='top', linespacing=0.65, rotation=37)

    main_ax.text(
        0.02, 0.79,
        "{} {}-mutated samples\n{:.1%} of {} cohort affected".format(
            len(use_mtree), args.gene,
            len(use_mtree) / len(base_cdict.get_samples()), args.cohort,
            ),
        size=11, va='bottom', transform=main_ax.transAxes
        )

    main_ax.set_xlabel("Amino Acid Position", size=15, weight='semibold')
    main_ax.set_ylabel("# of Mutated Samples", size=15, weight='semibold')
    main_ax.grid(linewidth=0.31)

    main_ax.set_xlim(loc_counts[0][0] - exn_pos / 217, exn_pos * 1.01)
    main_ax.set_ylim(max_count / -3.6, max_count * 1.07)
    main_ax.set_yticks([tck for tck in main_ax.get_yticks() if tck >= 0])

    # save the plot to file
    fig.savefig(os.path.join(
        plot_dir, "mut-lollipop_{}__{}.svg".format(args.cohort, args.gene)
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
                if 'Domain_' in lvls}
            }
        for (src, clf), outs in out_use.groupby(['Source', 'Classif'])
        }

    # load protein domain data, get location of local cache for TCGA data
    domn_dict = {
        domn: get_protein_domains(domain_dir, domn)
        for domn in {lvls.split('Domain_')[1].split('__')[0]
                     for lvls in reduce(or_,
                                        [set(cdict.keys()) - {'Loc'}
                                         for cdict in cdata_dict.values()])}
        }

    infer_dict = {
        (src, lvls, clf): pickle.load(bz2.BZ2File(os.path.join(
            base_dir, "{}__{}__samps-{}".format(src, args.cohort, ctf),
            "out-data__{}__{}.p.gz".format(lvls, clf)
            ), 'r'))['Infer']
        for (src, lvls, clf), ctf in out_use.iteritems()
        if lvls == 'Exon__Location__Protein' or 'Domain_' in lvls
        }
 
    out_dict = {
        (src, lvls, clf): pickle.load(bz2.BZ2File(os.path.join(
            base_dir, "{}__{}__samps-{}".format(src, args.cohort, ctf),
            "out-aucs__{}__{}.p.gz".format(lvls, clf)
            ), 'r'))
        for (src, lvls, clf), ctf in out_use.iteritems()
        if lvls == 'Exon__Location__Protein' or 'Domain_' in lvls
        }

    plot_mutation_lollipop(cdata_dict, domn_dict, args)


if __name__ == '__main__':
    main()

