
import os
import sys

if 'DATADIR' in os.environ:
    base_dir = os.path.join(os.environ['DATADIR'],
                            'HetMan', 'subvariant_isolate')
else:
    base_dir = os.path.dirname(__file__)

plot_dir = os.path.join(base_dir, 'plots', 'gene')
sys.path.extend([os.path.join(os.path.dirname(__file__), '../../..')])

from HetMan.experiments.utilities.load_input import load_firehose_cohort
from HetMan.experiments.subvariant_isolate import domain_dir
from HetMan.features.data.domains import get_protein_domains
from dryadic.features.mutations import MuType

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

plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'
plt.rcParams['axes.edgecolor']='white'

variant_mtypes = (
    ('Loss', MuType({('Scale', 'Copy'): {(
        'Copy', ('ShalDel', 'DeepDel')): None}})),
    ('Point', MuType({('Scale', 'Point'): None})),
    ('Gain', MuType({('Scale', 'Copy'): {(
        'Copy', ('ShalGain', 'DeepGain')): None}}))
    )

variant_clrs = {'Point': "0.59", 'Gain': "#2A941F", 'Loss': "#B2262D"}


def plot_mutation_lollipop(cdata, domain_data, args):
    fig, main_ax = plt.subplots(figsize=(10, 4))

    pnt_muts = cdata.train_mut['Point']
    samp_dict = {lbl: mtype.get_samples(cdata.train_mut)
                 for lbl, mtype in variant_mtypes}

    mut_counts = sorted([(int(loc), len(muts)) for loc, muts in pnt_muts
                         if loc != '.'],
                        key=itemgetter(0))

    mrks, stms, basl = main_ax.stem(*zip(*mut_counts))
    plt.setp(mrks, markersize=7, markeredgecolor='black', zorder=5)
    plt.setp(stms, linewidth=0.8, color='black', zorder=1)
    plt.setp(basl, linewidth=1.1, color='black', zorder=2)

    for loc, mut_count in mut_counts:
        if mut_count >= 20:
            mut_lbls = sorted(lbl for lbl, _ in pnt_muts[str(loc)])
            lbl_root = mut_lbls[0][2:-1]
 
            main_ax.text(loc + mut_counts[-1][0] / 109, mut_count,
                         lbl_root + "/".join(lbl.split(lbl_root)[-1]
                                             for lbl in mut_lbls),
                         size=11, ha='left', va='bottom')

            pie_ax = inset_axes(main_ax, width=0.57, height=0.57,
                                bbox_to_anchor=(loc, mut_count),
                                bbox_transform=main_ax.transData,
                                loc=4, axes_kwargs=dict(aspect='equal'),
                                borderpad=0)

            loc_samps = pnt_muts[str(loc)].get_samples()
            loc_ovlps = [len(loc_samps & samp_dict[lbl]) if lbl != 'Point'
                         else len(loc_samps
                                  - samp_dict['Gain'] - samp_dict['Loss'])
                         for lbl, _ in variant_mtypes]

            loc_croxs = [[[loc_ovlp, len(loc_samps - samp_dict[lbl])],
                          [len(samp_dict[lbl] - loc_samps),
                           len(cdata.samples - loc_samps - samp_dict[lbl])]]
                         if lbl != 'Point' else None
                         for loc_ovlp, (lbl, _) in zip(loc_ovlps,
                                                       variant_mtypes)]

            loc_tests = [(fisher_exact(loc_crox, alternative='less')[1],
                          fisher_exact(loc_crox, alternative='greater')[1])
                         if lbl != 'Point' else None
                         for loc_crox, (lbl, _) in zip(loc_croxs,
                                                       variant_mtypes)]

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

            pie_ptchs, pie_txts = pie_ax.pie(
                x=loc_ovlps, labels=loc_lbls, explode=[0.13, 0, 0.13],
                colors=[variant_clrs[lbl] for lbl, _ in variant_mtypes],
                labeldistance=0.47, wedgeprops=dict(alpha=0.73)
                )

            for i in range(len(pie_txts)):
                pie_txts[i].set_fontsize(7.9)
                pie_txts[i].set_horizontalalignment('center')


    max_count = max(count for _, count in mut_counts)
    gene_doms = domain_data[
        domain_data['Gene'] == cdata.gene_annot[args.gene]['Ens']]

    use_txs = set(gene_doms['Transcript'][
        gene_doms['DomainStart'] == gene_doms['DomainStart'].min()])
    use_txs &= set(gene_doms['Transcript'][
        gene_doms['DomainEnd'] == gene_doms['DomainEnd'].max()])
    gene_doms = gene_doms[gene_doms['Transcript'] == tuple(use_txs)[0]]

    prot_patches = []
    for _, (_, _, dom_id, dom_start, dom_end) in gene_doms.iterrows():
        prot_patches.append(Rect((dom_start, max_count * -0.13),
                                 dom_end - dom_start, max_count * 0.084,
                                 facecolor='red'))

        main_ax.text((dom_start + dom_end) / 2, max_count * -0.09, dom_id,
                     size=9, ha='center', va='center')

    main_ax.add_collection(PatchCollection(prot_patches,
                                           alpha=0.4, linewidth=0))

    main_ax.text(
        0.02, 0.41,
        "{} {}-mutated samples\n{:.1%} of {} cohort affected".format(
            len(pnt_muts), args.gene,
            len(pnt_muts) / len(cdata.samples), args.cohort,
            ),
        size=10, va='bottom', transform=main_ax.transAxes
        )

    main_ax.set_xlabel("Amino Acid Position", size=15, weight='semibold')
    main_ax.set_ylabel("# of Mutated Samples", size=15, weight='semibold')
    main_ax.grid(linewidth=0.31)
    main_ax.set_xlim(1, mut_counts[-1][0] * 1.04)
    main_ax.set_ylim(max_count / -7, max_count * 1.21)

    venn_ax = inset_axes(
        main_ax, width=2.19, height=1.31, loc=3,
        bbox_to_anchor=(mut_counts[-1][0] / 103, max_count * 0.67),
        bbox_transform=main_ax.transData, borderpad=0
        )

    v_plot = venn3([samp_dict[lbl] for lbl, _ in variant_mtypes],
                   ["Losses", "Mutations", "Gains"],
                   [variant_clrs[lbl] for lbl, _ in variant_mtypes],
                   alpha=0.73, ax=venn_ax)

    for i in range(len(v_plot.set_labels)):
        if v_plot.set_labels[i] is not None:
            v_plot.set_labels[i].set_fontsize(11)
    for i in range(len(v_plot.subset_labels)):
        if v_plot.subset_labels[i] is not None:
            v_plot.subset_labels[i].set_fontsize(10)

    fig.savefig(os.path.join(
        plot_dir, "mut-lollipop_{}__{}_domains-{}.png".format(
            args.cohort, args.gene, args.domains)
        ), dpi=250, format='png', bbox_inches='tight')

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

    cdata = load_firehose_cohort(args.cohort, [args.gene],
                                 ['Location', 'Protein'])
    domain_data = get_protein_domains(domain_dir, args.domains)

    plot_mutation_lollipop(cdata, domain_data, args)


if __name__ == '__main__':
    main()

