
import os
import sys

base_dir = os.path.join(os.environ['DATADIR'], 'HetMan', 'subvariant_tour')
sys.path.extend([os.path.join(os.path.dirname(__file__), '..', '..', '..')])
plot_dir = os.path.join(base_dir, 'plots', 'classif')

from HetMan.experiments.subvariant_tour import pnt_mtype
from HetMan.experiments.subvariant_tour.utils import RandomType
from HetMan.experiments.subvariant_tour.plot_gene import choose_cohort_colour
from dryadic.features.mutations import MuType

import argparse
from pathlib import Path
import bz2
import dill as pickle

import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'
plt.rcParams['axes.edgecolor']='white'


def plot_gene_results(auc_vals, conf_vals, pheno_dict, args):
    fig, ax = plt.subplots(figsize=(13, 8))

    clr_dict = dict()
    for (coh, gene), auc_vec in auc_vals.groupby(
        lambda x: (x[0], x[1].get_labels()[0])):

        if len(auc_vec) > 1 and auc_vec.max() > 0.68:
            base_mtype = MuType({('Gene', gene): pnt_mtype})
            base_indx = auc_vec.index.get_loc((coh, base_mtype))

            _, best_subtype = auc_vec[:base_indx].append(
                auc_vec[(base_indx + 1):]).idxmax()
            best_indx = auc_vec.index.get_loc((coh, best_subtype))

            if auc_vec[best_indx] > 0.68:
                conf_sc = np.greater.outer(conf_vals[coh, best_subtype],
                                           conf_vals[coh, base_mtype]).mean()

                if conf_sc > 0.6:
                    clr_dict[coh, gene] = choose_cohort_colour(coh)

                    base_size = 0.23 * np.mean(pheno_dict[coh][base_mtype])
                    best_prop = 0.23 * np.mean(
                        pheno_dict[coh][best_subtype]) / base_size

                    pie_ax = inset_axes(
                        ax, width=base_size ** 0.5, height=base_size ** 0.5,
                        bbox_to_anchor=(auc_vec[best_indx], conf_sc),
                        bbox_transform=ax.transData, loc=10,
                        axes_kwargs=dict(aspect='equal'), borderpad=0
                        )

                    pie_ax.pie(x=[best_prop, 1 - best_prop],
                               explode=[0.19, 0],
                               colors=[clr_dict[coh, gene] + (0.83,),
                                       clr_dict[coh, gene] + (0.23,)],
                               wedgeprops=dict(edgecolor='black',
                                               linewidth=4 / 13))

    ax.plot([0.65, 1.0005], [1, 1], color='black', linewidth=1.7, alpha=0.89)
    ax.plot([1, 1], [0.59, 1.0005], color='black', linewidth=1.7, alpha=0.89)

    ax.set_xlabel('AUC of Best Found Subgrouping',
                  size=23, weight='semibold')
    ax.set_ylabel('Down-Sampled AUC\nSuperiority Confidence',
                  size=23, weight='semibold')

    ax.tick_params(pad=5.3)
    ax.set_xlim([0.68, 1.002])
    ax.set_ylim([0.6, 1.005])

    fig.savefig(os.path.join(plot_dir, args.expr_source,
                             "{}__gene-results.svg".format(args.classif)),
                bbox_inches='tight', format='svg')

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        "Summarizes the enumeration and prediction results for a given "
        "classifier over a particular source of expression data."
        )

    parser.add_argument('expr_source',
                        help="a source of expression data", type=str)
    parser.add_argument('classif', help='a mutation classifier')

    args = parser.parse_args()
    os.makedirs(os.path.join(plot_dir, args.expr_source), exist_ok=True)

    out_datas = [
        out_file.parts[-2:] for out_file in Path(base_dir).glob(os.path.join(
            "{}__*__samps-*".format(args.expr_source),
            "out-conf__*__{}.p.gz".format(args.classif)
            ))
        ]

    out_use = pd.DataFrame([
        {'Cohort': out_data[0].split('__')[-2],
         'Samps': int(out_data[0].split("__samps-")[1]),
         'Levels': '__'.join(out_data[1].split('__')[1:-1])}
        for out_data in out_datas
        ]).groupby('Cohort').filter(
            lambda outs: ('Exon__Location__Protein' in set(outs.Levels)
                          and outs.Levels.str.match('Domain_').any())
        ).groupby(['Cohort', 'Levels'])['Samps'].min()

    out_use = out_use[~out_use.index.get_level_values('Cohort').isin(
        ['BRCA', 'BRCA_LumA'])]

    phn_dict = {coh: dict()
                for coh in set(out_use.index.get_level_values('Cohort'))}
    auc_dict = dict()
    conf_dict = dict()

    for (coh, lvls), ctf in out_use.iteritems():
        out_tag = "{}__{}__samps-{}".format(args.expr_source, coh, ctf)

        with bz2.BZ2File(
                os.path.join(base_dir, out_tag,
                             "out-pheno__{}__{}.p.gz".format(
                                 lvls, args.classif)),
                'r') as fl:
            phn_dict[coh].update(pickle.load(fl))

        with bz2.BZ2File(
                os.path.join(base_dir, out_tag,
                             "out-aucs__{}__{}.p.gz".format(
                                 lvls, args.classif)),
                'r') as fl:
            auc_vals = pickle.load(fl).Chrm

        auc_vals = auc_vals[[not isinstance(mtype, RandomType)
                             for mtype in auc_vals.index]]
        auc_vals.index = pd.MultiIndex.from_product([[coh], auc_vals.index],
                                                    names=('Cohort', 'Mtype'))
        auc_dict[coh, lvls] = auc_vals

        with bz2.BZ2File(
                os.path.join(base_dir, out_tag,
                             "out-conf__{}__{}.p.gz".format(
                                 lvls, args.classif)),
                'r') as fl:
            conf_vals = pickle.load(fl)['Chrm'].iloc[:, 0]

        conf_vals = conf_vals[[not isinstance(mtype, RandomType)
                               for mtype in conf_vals.index]]
        conf_vals.index = pd.MultiIndex.from_product(
            [[coh], conf_vals.index], names=('Cohort', 'Mtype'))
        conf_dict[coh, lvls] = conf_vals

    auc_vals = pd.concat(auc_dict.values())
    conf_vals = pd.concat(conf_dict.values())

    plot_gene_results(auc_vals, conf_vals, phn_dict, args)


if __name__ == "__main__":
    main()

