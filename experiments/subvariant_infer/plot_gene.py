
import os
import sys

if 'DATADIR' in os.environ:
    base_dir = os.path.join(os.environ['DATADIR'],
                            'HetMan', 'subvariant_infer')
else:
    base_dir = os.path.dirname(__file__)

plot_dir = os.path.join(base_dir, 'plots', 'gene')
sys.path.extend([os.path.join(os.path.dirname(__file__), '../../..')])

from HetMan.experiments.subvariant_infer.fit_infer import load_cohort_data
from HetMan.experiments.subvariant_infer import (
    domain_dir, variant_mtypes, variant_clrs, mcomb_clrs)
from dryadic.features.data.domains import get_protein_domains

import argparse
from operator import itemgetter
from scipy.stats import fisher_exact

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle as Rect
from matplotlib.collections import PatchCollection
from matplotlib_venn import venn3
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# make plots cleaner by turning off outer box, make background all white
plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'
plt.rcParams['axes.edgecolor']='white'


def plot_mutation_lollipop(cdata, domain_data, args):
    fig, main_ax = plt.subplots(figsize=(10, 4))

    # get tree of point mutations and the samples carrying
    # each of the major mutation types
    pnt_muts = cdata.train_mut['Point']
    samp_dict = {lbl: mtype.get_samples(cdata.train_mut)
                 for lbl, mtype in variant_mtypes}

    # get the number of samples with a point mutation at each amino acid
    mut_counts = sorted([(int(loc), len(muts)) for loc, muts in pnt_muts
                         if loc != '.'],
                        key=itemgetter(0))

    # create the mutation count lollipops and set the aesthetics of
    # the individual lollipop elements
    mrks, stms, basl = main_ax.stem(*zip(*mut_counts))
    plt.setp(mrks, markersize=5, markeredgecolor='black', zorder=5)
    plt.setp(stms, linewidth=0.8, color='black', zorder=1)
    plt.setp(basl, linewidth=1.1, color='black', zorder=2)

    # for each amino acid location with at least twenty mutated samples, get
    # the list of specific amino acid substitutions present at that location
    for loc, mut_count in mut_counts:
        if mut_count >= 20:
            mut_lbls = sorted(lbl for lbl, _ in pnt_muts[str(loc)])
            lbl_root = mut_lbls[0][2:-1]

            # create a label on the plot next to the head of the lollipop for
            # this point mutation location listing the aa substitutions
            main_ax.text(loc + mut_counts[-1][0] / 109, mut_count,
                         lbl_root + "/".join(lbl.split(lbl_root)[-1]
                                             for lbl in mut_lbls),
                         size=11, ha='left', va='bottom')

            # create a plotting space for a Venn diagram showing the overlap
            # between these point mutations and copy number alterations
            pie_ax = inset_axes(main_ax, width=0.57, height=0.57,
                                bbox_to_anchor=(loc, mut_count),
                                bbox_transform=main_ax.transData,
                                loc=4, axes_kwargs=dict(aspect='equal'),
                                borderpad=0)

            # get the number of samples with a point mutation at this location
            # that also have a gain or loss alteration, or neither
            loc_samps = pnt_muts[str(loc)].get_samples()
            loc_ovlps = [len(loc_samps & samp_dict[lbl]) if lbl != 'Point'
                         else len(loc_samps
                                  - samp_dict['Gain'] - samp_dict['Loss'])
                         for lbl, _ in variant_mtypes]

            # get the 2x2 tables of sample overlap for those with point
            # mutations and those with gain or loss alterations
            loc_croxs = [[[loc_ovlp, len(loc_samps - samp_dict[lbl])],
                          [len(samp_dict[lbl] - loc_samps),
                           len(cdata.samples - loc_samps - samp_dict[lbl])]]
                         if lbl != 'Point' else None
                         for loc_ovlp, (lbl, _) in zip(loc_ovlps,
                                                       variant_mtypes)]

            # test for statistically significant co-occurence or mutual
            # exclusivity between these point mutations and alterations
            loc_tests = [(fisher_exact(loc_crox, alternative='less')[1],
                          fisher_exact(loc_crox, alternative='greater')[1])
                         if lbl != 'Point' else None
                         for loc_crox, (lbl, _) in zip(loc_croxs,
                                                       variant_mtypes)]

            # create labels for sample ounts and significance for the overlap
            # Venn diagram 
            loc_lbls = [str(loc_ovlp) for loc_ovlp in loc_ovlps]
            for i, (loc_test, (lbl, _)) in enumerate(zip(loc_tests,
                                                         variant_mtypes)):
                if lbl != 'Point':

                    if loc_test[0] < 0.05:
                        loc_lbls[i] += '(-)'
                        if loc_test[0] < 0.001:
                            loc_lbls[i] += '**'
                        else:
                            loc_lbls[i] += '*'

                    if loc_test[1] < 0.05:
                        loc_lbls[i] += '(+)'
                        if loc_test[1] < 0.001:
                            loc_lbls[i] += '**'
                        else:
                            loc_lbls[i] += '*'

            # plot the overlap Venn diagram next to the head of the lollipop
            pie_ptchs, pie_txts = pie_ax.pie(
                x=loc_ovlps, labels=loc_lbls, explode=[0.13, 0, 0.13],
                colors=[variant_clrs[lbl] if lbl == 'Point'
                        else mcomb_clrs["Point+{}".format(lbl)]
                        for lbl, _ in variant_mtypes],
                labeldistance=0.47, wedgeprops=dict(alpha=0.71)
                )

            # adjust the properties of the Venn diagram's text annotation
            for i in range(len(pie_txts)):
                pie_txts[i].set_fontsize(7)
                pie_txts[i].set_horizontalalignment('center')

    gn_annot = cdata.gene_annot[args.gene]
    main_tx = {tx_id for tx_id, tx_annot in gn_annot['Transcripts'].items()
               if tx_annot['transcript_name'] == '{}-001'.format(args.gene)}

    prot_patches = []
    max_count = max(count for _, count in mut_counts)
    gene_doms = domain_data[(domain_data['Gene'] == gn_annot['Ens'])
                            & (domain_data['Transcript'].isin(main_tx))]

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
        exn_len = exn_annot['End'] - exn_annot['Start']

        if 'UTR' in exn_annot:
            for utr_annot in exn_annot['UTR']:
                exn_len -= utr_annot['End'] - utr_annot['Start']

        if exn_len > 0 and exn_pos <= mut_counts[-1][0]:
            exn_len /= 3

            exn_patches.append(Rect((exn_pos, max_count * -0.23),
                                    exn_len, max_count * 0.08,
                                    color='green'))

            main_ax.text(exn_pos + exn_len / 2, max_count * -0.196,
                         "{}/{}".format(i + 1, len(gn_annot['Exons'])),
                         size=min(
                             11, (531 * exn_len / mut_counts[-1][0]) ** 0.6),
                         ha='center', va='center')
            exn_pos += exn_len

    main_ax.add_collection(PatchCollection(
        exn_patches, alpha=0.4, linewidth=1.4, color='#002C91'))

    main_ax.text(exn_pos / -391, max_count * -0.05,
                 "{}\nDomains".format(args.domains), size=7,
                 ha='right', va='top', linespacing=0.65, rotation=37)

    main_ax.text(exn_pos / -391, max_count * -0.16,
                 "{}-001\nExons".format(args.gene), size=7,
                 ha='right', va='top', linespacing=0.65, rotation=37)

    main_ax.text(
        0.02, 0.34,
        "{} {}-mutated samples\n{:.1%} of {} cohort affected".format(
            len(pnt_muts), args.gene,
            len(pnt_muts) / len(cdata.samples), args.cohort,
            ),
        size=9, va='bottom', transform=main_ax.transAxes
        )

    main_ax.set_xlabel("Amino Acid Position", size=15, weight='semibold')
    main_ax.set_ylabel("# of Mutated Samples", size=15, weight='semibold')
    main_ax.grid(linewidth=0.31)

    main_ax.set_xlim(exn_pos / -519, exn_pos * 1.01)
    main_ax.set_ylim(max_count / -3.6, max_count * 1.21)
    main_ax.set_yticks([tck for tck in main_ax.get_yticks() if tck >= 0])

    venn_ax = inset_axes(
        main_ax, width=2.19, height=1.31, loc=3,
        bbox_to_anchor=(mut_counts[-1][0] / 103, max_count * 0.67),
        bbox_transform=main_ax.transData, borderpad=0
        )

    v_plot = venn3([samp_dict[lbl] for lbl, _ in variant_mtypes[::-1]],
                   ["Gains", "Point\nMutations", "Losses"],
                   [variant_clrs[lbl] for lbl, _ in variant_mtypes[::-1]],
                   alpha=0.71, ax=venn_ax)

    for i in range(len(v_plot.set_labels)):
        if v_plot.set_labels[i] is not None:
            v_plot.set_labels[i].set_fontsize(11)
    for i in range(len(v_plot.subset_labels)):
        if v_plot.subset_labels[i] is not None:
            v_plot.subset_labels[i].set_fontsize(10)

    # save the plot to file
    fig.savefig(os.path.join(
        plot_dir, "mut-lollipop_{}__{}_domains-{}.png".format(
            args.cohort, args.gene, args.domains)
        ), dpi=350, format='png', bbox_inches='tight')

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        "Plots the incidence and structure of the variants of a gene "
        "present in the samples of a particular cohort."
        )

    # create command line arguments
    parser.add_argument('cohort', help='a TCGA cohort')
    parser.add_argument('gene', help='a mutated gene')
    parser.add_argument('--domains', '-d', default='Pfam')

    # parse command line arguments, create directory where plots will be saved
    args = parser.parse_args()
    os.makedirs(plot_dir, exist_ok=True)

    # load protein domain data, get location of local cache for TCGA data
    domain_data = get_protein_domains(domain_dir, args.domains)
    cdata = load_cohort_data(base_dir,
                             args.cohort, args.gene, "Location__Protein")

    plot_mutation_lollipop(cdata, domain_data, args)


if __name__ == '__main__':
    main()

