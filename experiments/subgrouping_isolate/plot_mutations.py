
import os
import argparse
from pathlib import Path
import bz2
from operator import itemgetter
import re

import dill as pickle
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
from matplotlib.patches import Rectangle as Rect
from matplotlib.collections import PatchCollection

from ..utilities.mutations import (
    pnt_mtype, copy_mtype,
    gains_mtype, dels_mtype, ExMcomb, )
from dryadic.features.mutations import MuType
from dryadic.features.data.domains import get_protein_domains
from .data_dirs import domain_dir
from ..utilities.labels import get_fancy_label
from ..subvariant_test.utils import get_cohort_label
from ..utilities.colour_maps import form_clrs
from .plot_gene import choose_subtype_colour

# make plots cleaner by turning off outer box, make background all white
mpl.use('Agg')
plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'
plt.rcParams['axes.edgecolor']='white'


base_dir = os.path.join(os.environ['DATADIR'], 'HetMan',
                        'subgrouping_isolate')
plot_dir = os.path.join(base_dir, 'plots', 'mutations')


def get_pie_counts(base_mtype, use_mtree):
    all_type = MuType(use_mtree.allkey())

    pie_mtypes = [ExMcomb(copy_mtype, base_mtype),
                  ExMcomb(all_type, gains_mtype, base_mtype),
                  ExMcomb(all_type, dels_mtype, base_mtype)]

    return [len(mcomb.get_samples(use_mtree)) for mcomb in pie_mtypes]


def plot_lollipop(cdata_dict, domain_dict, args):
    fig, ax = plt.subplots(figsize=(9, 5))

    use_cdata = tuple(cdata_dict.values())[0]
    gn_annot = use_cdata.gene_annot[args.gene]
    loc_lvls = 'Gene', 'Scale', 'Copy', 'Consequence', 'Position', 'HGVSp'
    if loc_lvls not in use_cdata.mtrees:
        use_cdata.add_mut_lvls(loc_lvls)

    use_mtree = use_cdata.mtrees[loc_lvls][args.gene]
    mut_count = len(use_mtree.get_samples())
    var_count = len(use_mtree['Point'].get_samples())

    pie_clrs = [choose_subtype_colour(pnt_mtype),
                choose_subtype_colour(pnt_mtype | gains_mtype),
                choose_subtype_colour(pnt_mtype | dels_mtype)]

    pie_ax = ax.inset_axes(bounds=(0.05, 0.41, 0.27, 0.27))
    pie_ax.pie(x=get_pie_counts(pnt_mtype, use_mtree), colors=pie_clrs,
               labels=['point muts\nwithout CNAs',
                       'point & any gain', 'point & any loss'],
               labeldistance=1.21, autopct='%.0f', pctdistance=0.37,
               explode=[0, 0.19, 0.19], startangle=90,
               wedgeprops=dict(alpha=0.67), textprops=dict(size=9))

    loc_dict = {
        form: sorted([(int(loc.split('-')[0]), len(loc_muts.get_samples()))
                      for loc, loc_muts in form_muts
                      if loc.split('-')[0].isnumeric()],
                     key=itemgetter(0))
        for form, form_muts in use_mtree['Point']
        }

    # calculate the minimum and maximum amino acid positions for the
    # mutations to be plotted, as well as the most samples at any hotspot
    min_pos = min(pos for loc_counts in loc_dict.values()
                  for pos, _ in loc_counts if pos >= 0)
    max_pos = max(pos for loc_counts in loc_dict.values()
                  for pos, _ in loc_counts)
    max_count = max(count for loc_counts in loc_dict.values()
                    for _, count in loc_counts)

    pos_rng = max_pos - min_pos
    lgnd_ptchs = []

    for form, form_muts in use_mtree['Point']:
        if loc_dict[form]:
            form_mtype = MuType({
                ('Scale', 'Point'): {('Consequence', form): None}})

            lgnd_ptchs += [Patch(color=form_clrs[form], alpha=0.67,
                                 label=get_fancy_label(form_mtype))]
            mrks, stms, basl = ax.stem(*zip(*loc_dict[form]),
                                       use_line_collection=True)

            plt.setp(mrks, markersize=7, markeredgecolor='black',
                     markerfacecolor=form_clrs[form], zorder=5)
            plt.setp(stms, linewidth=0.8, color='black', zorder=1)
            plt.setp(basl, linewidth=1.1, color='black', zorder=2)

            for loc, loc_muts in form_muts:
                loc_size = len(loc_muts.get_samples())

                if loc != '-' and loc_size >= 10:
                    loc_int = int(loc.split('-')[0])

                    mut_lbls = sorted(
                        get_fancy_label(MuType({
                            ('Scale', 'Point'): {('HGVSp', lbl): None}}))
                        for lbl, _ in loc_muts
                        )

                    root_indx = re.match('[A-Z][0-9]+', mut_lbls[0]).span()[1]
                    lbl_root = mut_lbls[0][:root_indx]

                    if max(len(lbl) - len(lbl_root) for lbl in mut_lbls) > 4:
                        loc_lbl = '\n'.join(
                            [mut_lbls[0]] + [''.join([' ' * len(lbl_root) * 2,
                                                      lbl.split(lbl_root)[1]])
                                             for lbl in mut_lbls[1:]]
                            )

                    else:
                        loc_lbl = "/".join(
                            [mut_lbls[0]] + [lbl.split(lbl_root)[1]
                                             for lbl in mut_lbls[1:]]
                            )

                    ax.text(loc_int + pos_rng / 115,
                            loc_size + max_count / 151,
                            loc_lbl, size=8, ha='left', va='center')

                    loc_mtype = MuType({('Scale', 'Point'): {
                        ('Consequence', form): {('Position', loc): None}}})

                    pie_ax = ax.inset_axes(
                        bounds=(loc_int - pos_rng / 7, loc_size,
                                max_pos / 5, max_count / 5),
                        transform=ax.transData
                        )

                    pie_ax.pie(x=get_pie_counts(loc_mtype, use_mtree),
                               colors=pie_clrs, explode=[0, 0.23, 0.23],
                               startangle=90, wedgeprops=dict(alpha=0.67))

    use_tx = use_cdata._muts.loc[
        (use_cdata._muts.Gene == args.gene)
        & ~use_cdata._muts.Feature.isnull()
        ].Feature.unique()

    assert len(use_tx) == 1, (
        "Multiple transcripts detected in {} for {} !".format(args.cohort,
                                                              args.gene)
        )

    use_tx = use_tx[0]
    prot_patches = []
    min_pos = max(min_pos - max_pos / 13.1, 1)

    for i, (domn_lbl, domn_data) in enumerate(domain_dict.items()):
        tx_annot = gn_annot['Transcripts'][use_tx]

        gene_domns = domn_data[(domn_data.Gene == gn_annot['Ens'])
                               & (domn_data.Transcript == use_tx)]
        min_pos = max(
            min(min_pos, gene_domns.DomainStart.min() - max_pos / 9.7), 1)

        for domn_id, domn_start, domn_end in zip(gene_domns.DomainID,
                                                 gene_domns.DomainStart,
                                                 gene_domns.DomainEnd):
            prot_patches.append(Rect(
                (domn_start, -max_count * (0.15 + i * 0.11)),
                domn_end - domn_start, max_count * 0.09
                ))

            ax.text((domn_start + domn_end) / 2,
                    -max_count * (0.11 + i * 0.11),
                    domn_id, size=9, ha='center', va='center')

    ax.add_collection(PatchCollection(prot_patches, color='#D99100',
                                      alpha=0.4, linewidth=0))

    for i, domn_nm in enumerate(domain_dict):
        ax.text(min_pos, -max_count * (0.07 + i * 0.13),
                "{}\nDomains".format(domn_nm), size=8,
                ha='right', va='top', linespacing=0.71, rotation=37)

    ax.text(0.03, 0.98, args.gene, size=14, fontweight='semibold',
            ha='left', va='center', transform=ax.transAxes)

    ax.text(
        0.03, 0.941,
        "{} point mutants\n{} gain mutants\n{} loss mutants"
        "\n{:.1%} of {} affected".format(
            var_count, len(gains_mtype.get_samples(use_mtree)),
            len(dels_mtype.get_samples(use_mtree)),
            mut_count / len(use_cdata.get_samples()),
            get_cohort_label(args.cohort),
            ),
        size=12, ha='left', va='top', transform=ax.transAxes
        )

    # add the legend for the colour used for each form of mutation
    plt_lgnd = ax.legend(handles=lgnd_ptchs, frameon=False, fontsize=11,
                         ncol=2, loc=1, handletextpad=0.7,
                         bbox_to_anchor=(0.98, 1.02))
    ax.add_artist(plt_lgnd)

    ax.grid(linewidth=0.41, alpha=0.41)
    ax.set_xlabel("Amino Acid Position", size=17, weight='semibold')
    ax.set_ylabel("       # of Mutated Samples", size=17, weight='semibold')

    ax.set_xlim(min_pos, max_pos * 1.02)
    ax.set_ylim(-max_count * (0.05 + len(domain_dict) * 0.13),
                max_count * 17 / 11)
    ax.set_yticks([tck for tck in ax.get_yticks() if tck >= 0])

    fig.savefig(os.path.join(plot_dir, args.cohort,
                             "{}_lollipop.svg".format(args.gene)),
                bbox_inches='tight', format='svg')

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        'plot_mutations',
        description="Plots the incidence of a gene's variants in a cohort."
        )

    parser.add_argument('cohort', help="a tumour cohort")
    parser.add_argument('gene', help="a mutated gene")

    args = parser.parse_args()
    out_list = tuple(Path(base_dir).glob(
        os.path.join("*__{}".format(args.cohort), "out-conf__*__*__*.p.gz")))

    if len(out_list) == 0:
        raise ValueError("No completed experiments found for this "
                         "combination of parameters!")

    out_use = pd.DataFrame(
        [{'Source': out_file.parts[-2].split('__')[0],
          'Levels': '__'.join(out_file.parts[-1].split('__')[1:-2]),
          'Classif': out_file.parts[-1].split('__')[-1].split('.p.gz')[0],
          'File': out_file}
         for out_file in out_list]
        )

    cdata_dict = {
        (src, clf): None
        for src, clf in out_use.groupby(['Source', 'Classif']).groups
        }

    out_iter = out_use.groupby(['Source', 'Levels', 'Classif'])['File']
    for (src, lvls, clf), out_files in out_iter:
        for out_file in out_files:
            out_dir = os.path.join(base_dir, '__'.join([src, args.cohort]))
            out_tag = '__'.join(out_file.parts[-1].split('__')[1:])

            with bz2.BZ2File(Path(out_dir,
                                  '__'.join(["cohort-data", out_tag])),
                             'r') as f:
                new_cdata = pickle.load(f)

            if cdata_dict[src, clf] is None:
                cdata_dict[src, clf] = new_cdata
            else:
                cdata_dict[src, clf].merge(new_cdata, use_genes=[args.gene])

    assert len({cdata.data_hash()[1]
                for cdata in cdata_dict.values()}) == 1, (
                    "Inconsistent mutation data between different iterations "
                    "of the experiment!"
                    )

    domn_dict = {domn: get_protein_domains(domain_dir, domn)
                 for domn in ['Pfam', 'SMART']}
    os.makedirs(os.path.join(plot_dir, args.cohort), exist_ok=True)

    # create the plots
    plot_lollipop(cdata_dict, domn_dict, args)


if __name__ == '__main__':
    main()

