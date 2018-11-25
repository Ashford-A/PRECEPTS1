
import os
import sys

if 'DATADIR' in os.environ:
    base_dir = os.path.join(os.environ['DATADIR'],
                            'HetMan', 'gene_cluster')
else:
    base_dir = os.path.dirname(__file__)

plot_dir = os.path.join(base_dir, 'plots', 'model')
sys.path.extend([os.path.join(os.path.dirname(__file__), '../../..')])

from dryadic.features.mutations import MuType
from HetMan.experiments.gene_cluster import *
from HetMan.experiments.gene_cluster.setup_cluster import get_cohort_data

import numpy as np
import argparse
import dill as pickle

from operator import itemgetter, or_
from functools import reduce

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'
plt.rcParams['axes.edgecolor']='white'

mut_sets = [('Point', "Mutations"),
            ('Hom', "Mutations\n+ Homozygous Alterations"),
            ('Het', "Mutations\n+ All Alterations")]
feat_sets = [('All', "all gene features"),
             ('Chr', "gene features\nnot on same chr")]

mut_clrs = {'HomDel': '#BA001D', 'HetDel': '#F23654', 'None': '0.47',
            'HomGain': '#0E2A85', 'HetGain': '#3854B2'}
mut_edges = {'HomDel': '#920017', 'HetDel': 'none', 'None': 'none',
             'HomGain': '#091F69', 'HetGain': 'none'}


def plot_mutations(model_data, args, cdata):
    fig, axarr = plt.subplots(figsize=(13, 9), nrows=2, ncols=3)

    for i, (mut_set, mut_lbl) in enumerate(mut_sets):
        for j, (feat_set, feat_lbl) in enumerate(feat_sets):
            mut_samps = cdata.train_mut[args.gene]['Point']

            if mut_set != 'Point':
                mut_samps |= hom_mtype.get_samples(cdata.train_mut[args.gene])
            if mut_set == 'Het':
                mut_samps |= het_mtype.get_samples(cdata.train_mut[args.gene])

            copy_phenos = {
                lbl: np.array(cdata.train_pheno(
                    MuType({('Gene', args.gene): mtype}), mut_samps))
                for lbl, mtype in copy_mtypes.items()
                }

            copy_phenos['None'] = ~reduce(or_, copy_phenos.values())
            pnt_phn = np.array(cdata.train_pheno(
                MuType({('Gene', args.gene): pnt_mtype}), mut_samps))

            for lbl, pheno in copy_phenos.items():
                axarr[j, i].scatter(
                    model_data[mut_set]['Mut'][feat_set][pheno & pnt_phn, 0],
                    model_data[mut_set]['Mut'][feat_set][pheno & pnt_phn, 1],
                    c=mut_clrs[lbl], marker='o', linewidths=1.2,
                    s=19, alpha=0.41, edgecolors=mut_edges[lbl]
                    )

                axarr[j, i].scatter(
                    model_data[mut_set]['Mut'][feat_set][pheno & ~pnt_phn, 0],
                    model_data[mut_set]['Mut'][feat_set][pheno & ~pnt_phn, 1],
                    c=mut_clrs[lbl], marker='s', linewidths=1.2,
                    s=23, alpha=0.37, edgecolors=mut_edges[lbl]
                    )

            if i == 0:
                axarr[j, i].set_ylabel(feat_lbl, size=21, weight='semibold')
            if j == 0:
                axarr[j, i].set_title(mut_lbl, size=19, weight='semibold')

    for ax in axarr.reshape(-1):
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    ax.legend(
        [Line2D([], [], marker='o', linestyle='None', markersize=19,
                alpha=0.41, markerfacecolor='0.43', markeredgecolor='none'),
         Line2D([], [], marker='s', linestyle='None', markersize=23,
                alpha=0.37, markerfacecolor='0.43', markeredgecolor='none'),
         Line2D([], [], marker='o', linestyle='None', markersize=21,
                alpha=0.41, markerfacecolor=mut_clrs['HomDel'],
                markeredgecolor='none'),
         Line2D([], [], marker='o', linestyle='None', markersize=21,
                alpha=0.41, markerfacecolor=mut_clrs['HomGain'],
                markeredgecolor='none'),
         Line2D([], [], marker='o', linestyle='None', markersize=21,
                alpha=0.41, markerfacecolor=mut_clrs['HomDel'],
                markeredgecolor=mut_clrs['HomDel'], markeredgewidth=1.7),
         Line2D([], [], marker='o', linestyle='None', markersize=21,
                alpha=0.41, markerfacecolor=mut_clrs['HomGain'],
                markeredgecolor=mut_clrs['HomGain'], markeredgewidth=1.7)],
        [args.gene + lbl for lbl in [" mutant samples", " wild-type samples",
                                     " shallow dels", " shallow amps",
                                     " deep dels", " deep amps"]],
        fontsize=19, ncol=3, bbox_to_anchor=(1.07, 0.)
        ).get_frame().set_linewidth(0.0)

    fig.savefig(os.path.join(plot_dir, args.expr_source,
                             "{}__samps-{}".format(args.cohort,
                                                   args.samp_cutoff),
                             "base-clustering_{}__{}.png".format(args.gene,
                                                                 args.model)),
                dpi=300, bbox_inches='tight')

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Plot experiment results for given mutation classifier.')

    parser.add_argument('expr_source', type=str,
                        choices=list(expr_sources.keys()),
                        help="which TCGA expression data source to use")

    parser.add_argument('cohort', help='a TCGA cohort')
    parser.add_argument('gene', help='a mutated gene')
    parser.add_argument('model', help='a clustering algorithm',
                        choices=['PCA', 't-SNE', 'UMAP'])
    parser.add_argument('--samp_cutoff', default=100)

    # parse command line arguments, create directory where plots will be saved
    args = parser.parse_args()
    cdata = get_cohort_data(args.expr_source, args.cohort)
    os.makedirs(
        os.path.join(plot_dir, args.expr_source,
                     "{}__samps-{}".format(args.cohort, args.samp_cutoff)),
        exist_ok=True
        )

    genes_list = pickle.load(
        open(os.path.join(base_dir, "setup",
                          "genes-list_{}__{}__samps-{}.p".format(
                              args.expr_source, args.cohort,
                              args.samp_cutoff
                            )),
             'rb')
        )

    gene_indx = {
        (i, mut_set) for i, (mut_set, mut_gene) in enumerate(genes_list)
        if mut_gene == args.gene
        }

    out_dir = os.path.join(base_dir, "output", args.expr_source,
                           "{}__samps-{}".format(args.cohort, args.samp_cutoff))
    out_files = sorted([(fl, int(fl.split('out_task-')[1].split('.p')[0]))
                        for fl in os.listdir(out_dir) if 'out_task-' in fl],
                       key=itemgetter(1))

    model_data = {
        mut_set: pickle.load(
            open(os.path.join(out_dir,
                              "out_task-{}.p".format(i % len(out_files))),
                 'rb')
            )[mut_set, args.gene][args.model]
        for i, mut_set in gene_indx
        }

    plot_mutations(model_data, args, cdata)
    

if __name__ == '__main__':
    main()

