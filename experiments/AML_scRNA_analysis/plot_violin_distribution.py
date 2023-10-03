
from ..utilities.mutations import pnt_mtype, copy_mtype, RandomType
from dryadic.features.mutations import MuType

#from ..subgrouping_test import base_dir
from ..AML_scRNA_analysis import base_dir
from .utils import filter_mtype
#from .plot_ccle import load_response_data
from .plot_mutations import recurse_labels

from ..utilities.colour_maps import variant_clrs
from ..utilities.misc import get_label, get_subtype, choose_label_colour
from ..utilities.labels import get_fancy_label, get_cohort_label
from ..utilities.label_placement import place_scatter_labels

import os
import argparse
from pathlib import Path
import bz2
import dill as pickle

import numpy as np
import pandas as pd

from itertools import combinations as combn
from scipy.stats import spearmanr

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Added by Andrew on 7/12/2023 to point the script to the correct run location. Instead of "base_dir", it was saved in "temp_dir" global variable
# base_dir points the script toward outputs in the temporary directory location where the script stored the intermediate files, it requires the creation of
# a plots/cluster directory within, for instance "dryads-research/AML_scRNA_analysis/default__default/
# For example: dryads-research/AML_scRNA_analysis/default__default/plots/cluster
#base_dir = '/home/groups/precepts/ashforda/single-cell_precepts/custom_precepts_inputs/AML556-D0/Temp_Files/dryads-research/AML_scRNA_analysis/default__default'
#base_dir = '/home/groups/precepts/ashforda/single-cell_precepts/custom_precepts_inputs/AML556-D0/dryads-research/AML_scRNA_analysis'
base_dir = '/home/groups/precepts/ashforda/single-cell_precepts/custom_precepts_inputs/vanGalen_D0_AML_samples_and_4_healthy_BM_samples/Temp_Files/dryads-research/AML_scRNA_analysis/default__default'

mpl.use('Agg')
plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'
plot_dir = os.path.join(base_dir, 'plots', 'violin_distribution')


def plot_remainder_scores(pred_mat, auc_vals, pheno_dict, cdata, args):
    fig, axarr = plt.subplots(figsize=(0.5 + pred_mat.shape[0] * 1.7, 7),
                              nrows=1, ncols=pred_mat.shape[0])

    base_mtype = MuType({('Gene', args.gene): pnt_mtype})
    if pred_mat.shape[0] == 1:
        axarr = [axarr]

    for i, (ax, plt_mtype) in enumerate(zip(axarr, pred_mat.index)):
        use_lvls = cdata.find_pheno(plt_mtype)
        use_mtree = cdata.mtrees[use_lvls][args.gene]['Point']
        leaf_count = len(MuType(use_mtree.allkey()).leaves())

        sns.violinplot(x=pred_mat.loc[plt_mtype][~pheno_dict[base_mtype]],
                       ax=ax, palette=[variant_clrs['WT']], inner=None,
                       orient='v', linewidth=0, cut=0, width=0.89)
        sns.violinplot(x=pred_mat.loc[plt_mtype][pheno_dict[plt_mtype]],
                       ax=ax, palette=[variant_clrs['Point']], inner=None,
                       orient='v', linewidth=0, cut=0, width=0.89)

        rest_stat = pheno_dict[base_mtype] & ~pheno_dict[plt_mtype]
        if rest_stat.sum() > 10:
            sns.violinplot(x=pred_mat.loc[plt_mtype][rest_stat],
                           ax=ax, palette=['none'], inner=None, orient='v',
                           linewidth=1.7, cut=0, width=0.89)

        else:
            ax.scatter(np.random.randn(rest_stat.sum()) / 7.3,
                       pred_mat.loc[plt_mtype][rest_stat],
                       facecolor='none', s=31, alpha=0.53,
                       edgecolors='black', linewidth=0.9)

        ax.get_children()[0].set_alpha(0.41)
        ax.get_children()[1].set_alpha(0.41)
        ax.get_children()[2].set_facecolor((1, 1, 1, 0))
        ax.get_children()[2].set_edgecolor((0, 0, 0, 0.47))

        tree_ax = inset_axes(ax, width='100%', height='100%',
                             bbox_to_anchor=(0.03, 0.89, 0.94, 0.09),
                             bbox_transform=ax.transAxes, borderpad=0)
        tree_ax.axis('off')
        tree_mtype = get_subtype(get_subtype(plt_mtype))

        tree_ax = recurse_labels(tree_ax, use_mtree, (0, leaf_count),
                                 len(use_lvls) - 2, leaf_count,
                                 clr_mtype=tree_mtype, add_lbls=False,
                                 mut_clr=variant_clrs['Point'])

        mtype_lbl = get_fancy_label(get_subtype(plt_mtype),
                                    pnt_link='\nor\n', phrase_link='\n')
        ax.text(0.5, 1.01, mtype_lbl,
                size=8, ha='center', va='bottom', transform=ax.transAxes)

        ylims = ax.get_ylim()
        ygap = (ylims[1] - ylims[0]) / 7
        ax.set_ylim([ylims[0], ylims[1] + ygap])
        ax.set_yticklabels([])

        if i == 0:
            ax.set_ylabel("Subgrouping Inferred Score",
                          size=21, weight='semibold')
        else:
            ax.set_ylabel('')

    plt.savefig(
        os.path.join(plot_dir,
                     '__'.join([args.expr_source, args.cohort]), args.gene,
                     "remainder-scores_{}.svg".format(args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()

def main():
    '''
    parser = argparse.ArgumentParser(
        "Plots the relationships between the scores inferred by a classifier "
        "for the subgroupings enumerated for a particular gene in a cohort."
        )

    parser.add_argument('expr_source',
                        help="a source of expression data", type=str)
    parser.add_argument('cohort', help="a TCGA cohort", type=str)
    parser.add_argument('gene', help="a mutated gene", type=str)
    parser.add_argument('classif', help="a mutation classifier", type=str)

    args = parser.parse_args()
    '''

    ########## Modified code for scAML violin distribution plots ##########

    parser = argparse.ArgumentParser(
        'plot_violin_distribution',
        description="Plots the mutation score distributions between mutant and WT samples."
        )
    
    parser.add_argument('classif')
    parser.add_argument('--feats_file')
    parser.add_argument('--comp_files', nargs='+')
    parser.add_argument('--seed', type=int, default=9087)

    args = parser.parse_args()
    np.random.seed(args.seed)

    ########## Modified code for scAML violin distribution plots ##########

    # I think this code is to look at multiple different runs over transfer cohorts
    '''
    out_datas = [
        out_file.parts[-2:] for out_file in Path(base_dir).glob(os.path.join(
            "{}__{}__samps-*".format(args.expr_source, args.cohort),
            "out-trnsf__*__{}.p.gz".format(args.classif)
            ))
        ]

    out_list = pd.DataFrame([{'Samps': int(out_data[0].split('__samps-')[1]),
                              'Levels': '__'.join(out_data[1].split(
                                  'out-trnsf__')[1].split('__')[:-1])}
                             for out_data in out_datas])

    if out_list.shape[0] == 0:
        raise ValueError("No experiment output found for these parameters!")

    out_use = out_list.groupby('Levels')['Samps'].min()
    if 'Consequence__Exon' not in out_use.index:
        raise ValueError("Cannot compare coefficients until this experiment "
                         "is run with mutation levels `Consequence__Exon` "
                         "which tests genes' base mutations!")

    os.makedirs(os.path.join(plot_dir,
                             '__'.join([args.expr_source, args.cohort])),
                exist_ok=True)


    cdata = None
    pred_dict = dict()
    phn_dict = dict()
    auc_dict = dict()

    for lvls, ctf in out_use.iteritems():
        out_tag = "{}__{}__samps-{}".format(
            args.expr_source, args.cohort, ctf)
    '''
    # Note: In the original code there is another indent in the code below
        
    # The following code block opens cohort-data__{}__{}.p.gz
    '''
    with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                  "cohort-data__{}__{}.p.gz".format(
                                      lvls, args.classif)),
                     'r') as f:
        new_cdata = pickle.load(f)
    '''

    # How to open this using scAML results
    with bz2.BZ2File(os.path.join(base_dir, args.classif,
                                  "cohort-data.p.gz"),
                     'r') as f:
        new_cdata = pickle.load(f)

    if cdata is None:
        cdata = new_cdata
    else:
        cdata.merge(new_cdata, use_genes=[args.gene])

    # The following code block opens out-pred__{}__{}.p.gz
    '''
    with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                  "out-pred__{}__{}.p.gz".format(
                                      lvls, args.classif)),
                     'r') as f:
        pred_data = pickle.load(f)
    '''

    # How to open this using scAML results
    with bz2.BZ2File(os.path.join(base_dir, args.classif,
                                  "out-sc.p.gz"),
                     'r') as f:
        pred_data = pickle.load(f)
    
    '''
    pred_dict[lvls] = pred_data.loc[[
        mtype for mtype in pred_data.index
        if (not isinstance(mtype, RandomType)
            and filter_mtype(mtype, args.gene))
        ]]
    
    '''
    
    # The following code block opens out-pheno__{}__{}.p.gz
    '''
    with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                  "out-pheno__{}__{}.p.gz".format(
                                      lvls, args.classif)),
                     'r') as f:
        phn_data = pickle.load(f)
    '''

    # How to open out-pheno file from scAML results
    with bz2.BZ2File(os.path.join(base_dir, args.classif,
                                  "out-pheno.p.gz"),
                     'r') as f:
        phn_data = pickle.load(f)

    phn_dict.update({mtype: phn for mtype, phn in phn_data.items()
                     if filter_mtype(mtype, args.gene)})

    # The following code block opens out-aucs__{}__{}.p.gz
    '''
    with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                  "out-aucs__{}__{}.p.gz".format(
                                      lvls, args.classif)),
                     'r') as f:
        auc_data = pickle.load(f)['mean']
    '''

    # How to open the out-aucs file using the scAML results
    with bz2.BZ2File(os.path.join(base_dir, args.classif,
                                  "out-aucs.p.gz"),
                     'r') as f:
        auc_data = pickle.load(f)

    auc_dict[lvls] = auc_data[[filter_mtype(mtype, args.gene)
                               for mtype in auc_data.index]]

pred_df = pd.concat(pred_dict.values())
if pred_df.shape[0] == 0:
    raise ValueError(
        "No classification tasks found for gene `{}`!".format(args.gene))

auc_vals = pd.concat(auc_dict.values())

os.makedirs(os.path.join(plot_dir,
                         '__'.join([args.expr_source, args.cohort]),
                         args.gene),
            exist_ok=True)

base_mtype = MuType({('Gene', args.gene): pnt_mtype})
wt_phn = ~phn_dict[base_mtype]
wt_preds = pred_df.loc[base_mtype][wt_phn].apply(np.mean)

ortho_dict = {mtype: mtype.get_sorted_levels()
              for mtype, auc_val in auc_vals.iteritems()
              if (not isinstance(mtype, RandomType)
                  and auc_val >= 0.7 and mtype != base_mtype
                  and (get_subtype(mtype) & copy_mtype).is_empty())}

if ortho_dict:
    pred_vals = pred_df.loc[ortho_dict].applymap(np.mean)

    corr_dict = {mtype: spearmanr(pred_vals.loc[mtype][wt_phn],
                                  wt_preds).correlation
                 for mtype in ortho_dict}

    divg_list = pd.Series({mtype: (auc_vals[mtype] - 0.5) * (1 - corr_val)
                           for mtype, corr_val in corr_dict.items()})

    plot_remainder_scores(
        pred_vals.loc[divg_list.sort_values().index[:10]],
        auc_vals, phn_dict, cdata, args
        )


if __name__ == '__main__':
    main()
