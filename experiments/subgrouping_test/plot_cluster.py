
from ..subgrouping_test import base_dir
from ..utilities.transformers import OmicPCA, OmicTSNE, OmicUMAP
from ..utilities.data_dirs import vep_cache_dir
from ..utilities.labels import get_cohort_label
from ...features.cohorts.utils import load_cohort, list_cohort_subtypes

import os
import argparse
from pathlib import Path
import bz2
import dill as pickle

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

mpl.use('Agg')
plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'
plot_dir = os.path.join(base_dir, 'plots', 'cluster')

clust_algs = {'PCA': OmicPCA(), 'tSNE': OmicTSNE(), 'UMAP': OmicUMAP()}


def plot_clustering(trans_expr, subt_data, cdata, args, pca_comps=(0, 1)):
    fig, ax = plt.subplots(figsize=(9, 8))

    plt_subts = subt_data.unique()
    subt_clrs = dict(zip(
        plt_subts, sns.color_palette('bright', n_colors=len(plt_subts))))

    trans_expr = trans_expr[:, np.array(pca_comps)]
    subt_stat = np.array([subt_data[samp]
                          if samp in subt_data.index else 'Not Available'
                          for samp in cdata.train_data(None)[0].index])

    lgnd_lbls = []
    lgnd_marks = []

    for sub_type in sorted(set(subt_stat)):
        subt_indx = subt_stat == sub_type

        if sub_type == 'Not Available':
            subt_clr = '0.53'
        else:
            subt_clr = subt_clrs[sub_type]

        ax.scatter(trans_expr[subt_indx, 0], trans_expr[subt_indx, 1],
                   marker='o', s=31, c=[subt_clr],
                   alpha=0.27, edgecolor='none')

        lgnd_lbls += ["{} ({})".format(sub_type, np.sum(subt_indx))]
        lgnd_marks += [Line2D([], [], marker='o', linestyle='None',
                              markersize=23, alpha=0.43,
                              markerfacecolor=subt_clr,
                              markeredgecolor='none')]

    ax.set_xlabel("{} Component {}".format(args.transform, pca_comps[0] + 1),
                  size=17, weight='semibold')
    ax.set_xticklabels([])

    ax.set_ylabel("{} Component {}".format(args.transform, pca_comps[1] + 1),
                  size=17, weight='semibold')
    ax.set_yticklabels([])

    ax.legend(lgnd_marks, lgnd_lbls, bbox_to_anchor=(0.5, -0.05),
              frameon=False, fontsize=21, ncol=3, loc=9, handletextpad=0.3)
    ax.grid(alpha=0.41, linewidth=0.9)

    ax.set_title(' - '.join([get_cohort_label(args.cohort),
                             args.expr_source]),
                 size=23, weight='semibold')

    fig.savefig(
        os.path.join(plot_dir, "{}__{}__{}_comps-{}_{}.svg".format(
            args.expr_source, args.cohort, args.transform,
            pca_comps[0], pca_comps[1]
            )),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        'plot_cluster',
        description="Plots the output of unsupervised learning on a cohort."
        )

    parser.add_argument('expr_source', help="a source of expression datasets")
    parser.add_argument('cohort', help="a tumour sample -omic dataset")

    parser.add_argument('transform', type=str,
                        choices=list(clust_algs.keys()),
                        help='an unsupervised learning method')

    parser.add_argument('--seed', type=int, default=9087)
    args = parser.parse_args()
    np.random.seed(args.seed)

    out_datas = [
        out_file.parts[-2:] for out_file in Path(base_dir).glob(os.path.join(
            "{}__{}__samps-*".format(args.expr_source, args.cohort),
            "out-trnsf__*__*.p.gz"
            ))
        ]

    out_list = pd.DataFrame([
        {'Samps': int(out_data[0].split('__samps-')[1]),
         'Levels': '__'.join(out_data[1].split(
             'out-trnsf__')[1].split('__')[:-1]),
         'Classif': out_data[1].split('__')[-1].split('.p.gz')[0]}
        for out_data in out_datas
        ])

    if out_list.shape[0] > 0:
        out_use = out_list.groupby(['Levels', 'Classif'])['Samps'].min()
        cdata = None

        for (lvls, clf), ctf in out_use.iteritems():
            out_tag = "{}__{}__samps-{}".format(
                args.expr_source, args.cohort, ctf)

            with bz2.BZ2File(os.path.join(
                    base_dir, out_tag,
                    "cohort-data__{}__{}.p.gz".format(lvls, clf)
                    ), 'r') as f:
                new_cdata = pickle.load(f)

            if cdata is None:
                cdata = new_cdata
            else:
                cdata.merge(new_cdata)

    else:
        cdata = load_cohort(args.cohort, args.expr_source,
                            ('Consequence', 'Exon'), vep_cache_dir)

    trans_expr = clust_algs[args.transform].fit_transform_coh(cdata)
    os.makedirs(plot_dir, exist_ok=True)
    type_dict = list_cohort_subtypes(args.cohort.split('_')[0])

    if type_dict:
        subt_data = pd.concat([pd.Series(subt, index=smps)
                               for subt, smps in type_dict.items()])
    else:
        subt_data = pd.Series({smp: 'Not Available'
                               for smp in cdata.get_samples()})

    plot_clustering(trans_expr.copy(), subt_data, cdata, args)


if __name__ == "__main__":
    main()

