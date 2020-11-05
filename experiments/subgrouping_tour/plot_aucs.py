
from dryadic.features.mutations import MuType
from ..subgrouping_tour import base_dir
from ..utilities.metrics import calc_conf
from ..utilities.misc import choose_label_colour
from ..utilities.labels import get_fancy_label
from ..utilities.label_placement import place_scatter_labels

import os
import argparse
import bz2
import dill as pickle
import numpy as np
from sklearn.metrics import average_precision_score as aupr_score

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

mpl.use('Agg')
plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'
plot_dir = os.path.join(base_dir, 'plots', 'aucs')


def plot_sub_comparisons(auc_vals, pheno_dict, conf_vals, args):
    fig, ax = plt.subplots(figsize=(10.3, 11))

    plot_dict = dict()
    line_dict = dict()
    plt_min = 0.57

    # for each gene whose mutations were tested, pick a random colour
    # to use for plotting the results for the gene
    for gene, auc_vec in auc_vals.groupby(
            lambda mtype: tuple(mtype.label_iter())[0]):

        # if there were subgroupings tested for the gene, find the results
        # for the mutation representing all point mutations for this gene...
        if len(auc_vec) > 1:
            base_mtype = MuType({('Gene', gene): None})

            base_indx = auc_vec.index.get_loc(base_mtype)
            best_subtype = auc_vec[:base_indx].append(
                auc_vec[(base_indx + 1):]).idxmax()

            # if the AUC for the optimal subgrouping is good enough, plot it
            # against the AUC for all point mutations of the gene...
            if auc_vec[best_subtype] > 0.6:
                auc_tupl = auc_vec[base_mtype], auc_vec[best_subtype]
                line_dict[auc_tupl] = dict(c=choose_label_colour(gene))

                base_size = np.mean(pheno_dict[base_mtype])
                plt_size = 0.07 * base_size ** 0.5
                plot_dict[auc_tupl] = [plt_size, ('', '')]
                plt_min = min(plt_min, auc_vec[base_indx] - 0.053,
                              auc_vec[best_subtype] - 0.029)

                best_prop = np.mean(pheno_dict[best_subtype]) / base_size
                conf_sc = calc_conf(conf_vals[best_subtype],
                                    conf_vals[base_mtype])

                # ...and if we are sure that the optimal subgrouping AUC is
                # better than the point mutation AUC then add a label with the
                # gene name and a description of the best found subgrouping...
                if conf_sc > 0.8:
                    plot_dict[auc_tupl][1] = gene, get_fancy_label(
                        tuple(best_subtype.subtype_iter())[0][1],
                        pnt_link='\nor ', phrase_link=' '
                        )

                # ...if we are not sure but the respective AUCs are still
                # pretty great then add a label with just the gene name...
                elif auc_tupl[0] > 0.7 or auc_tupl[1] > 0.7:
                    plot_dict[auc_tupl][1] = gene, ''

                auc_bbox = (auc_tupl[0] - plt_size / 2,
                            auc_tupl[1] - plt_size / 2, plt_size, plt_size)

                pie_ax = inset_axes(
                    ax, width='100%', height='100%',
                    bbox_to_anchor=auc_bbox, bbox_transform=ax.transData,
                    axes_kwargs=dict(aspect='equal'), borderpad=0
                    )

                pie_ax.pie(x=[best_prop, 1 - best_prop],
                           colors=[line_dict[auc_tupl]['c'] + (0.77,),
                                   line_dict[auc_tupl]['c'] + (0.29,)],
                           explode=[0.29, 0], startangle=90)

    plt_lims = plt_min, 1 + (1 - plt_min) / 61
    ax.grid(alpha=0.41, linewidth=0.9)

    ax.plot(plt_lims, [0.5, 0.5],
            color='black', linewidth=1.3, linestyle=':', alpha=0.71)
    ax.plot([0.5, 0.5], plt_lims,
            color='black', linewidth=1.3, linestyle=':', alpha=0.71)

    ax.plot(plt_lims, [1, 1], color='black', linewidth=1.9, alpha=0.89)
    ax.plot([1, 1], plt_lims, color='black', linewidth=1.9, alpha=0.89)
    ax.plot(plt_lims, plt_lims,
            color='#550000', linewidth=2.1, linestyle='--', alpha=0.41)

    ax.set_xlabel("Accuracy of Gene-Wide Classifier",
                  size=23, weight='semibold')
    ax.set_ylabel("Accuracy of Best Subgrouping Classifier",
                  size=23, weight='semibold')

    if plot_dict:
        lbl_pos = place_scatter_labels(plot_dict, ax,
                                       plt_lims=[[plt_min + 0.01, 0.99]] * 2,
                                       line_dict=line_dict)

    ax.set_xlim(plt_lims)
    ax.set_ylim(plt_lims)

    plt.savefig(
        os.path.join(plot_dir, '__'.join([args.expr_source, args.cohort]),
                     "sub-comparisons_{}.svg".format(args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_aupr_comparisons(pred_df, pheno_dict, auc_vals, conf_vals, args):
    fig, (base_ax, subg_ax) = plt.subplots(figsize=(17, 8), nrows=1, ncols=2)

    plot_dicts = {'Base': dict(), 'Subg': dict()}
    line_dicts = {'Base': dict(), 'Subg': dict()}
    plt_max = 0.53

    for gene, auc_vec in auc_vals.groupby(
            lambda mtype: tuple(mtype.label_iter())[0]):

        if len(auc_vec) > 1:
            base_mtype = MuType({('Gene', gene): None})

            base_indx = auc_vec.index.get_loc(base_mtype)
            best_subtype = auc_vec[:base_indx].append(
                auc_vec[(base_indx + 1):]).idxmax()

            if auc_vec[best_subtype] > 0.6:
                base_size = np.mean(pheno_dict[base_mtype])
                plt_size = 0.07 * base_size ** 0.5
                best_prop = np.mean(pheno_dict[best_subtype]) / base_size

                base_infr = pred_df.loc[base_mtype].apply(np.mean)
                best_infr = pred_df.loc[best_subtype].apply(np.mean)

                base_auprs = (aupr_score(pheno_dict[base_mtype], base_infr),
                              aupr_score(pheno_dict[base_mtype], best_infr))
                subg_auprs = (aupr_score(pheno_dict[best_subtype], base_infr),
                              aupr_score(pheno_dict[best_subtype], best_infr))

                conf_sc = calc_conf(conf_vals[best_subtype],
                                    conf_vals[base_mtype])

                base_lbl = '', ''
                subg_lbl = '', ''
                min_diff = np.log2(1.25)

                mtype_lbl = get_fancy_label(
                    tuple(best_subtype.subtype_iter())[0][1],
                    pnt_link='\nor ', phrase_link=' '
                    )

                if conf_sc > 0.9:
                    base_lbl = gene, mtype_lbl
                    subg_lbl = gene, mtype_lbl

                elif (auc_vec[base_indx] > 0.75
                        or auc_vec[best_subtype] > 0.75):
                    base_lbl = gene, ''
                    subg_lbl = gene, ''

                elif auc_vec[base_indx] > 0.6 or auc_vec[best_subtype] > 0.6:
                    if abs(np.log2(base_auprs[1] / base_auprs[0])) > min_diff:
                        base_lbl = gene, ''
                    if abs(np.log2(subg_auprs[1] / subg_auprs[0])) > min_diff:
                        subg_lbl = gene, ''

                for lbl, auprs, mtype_lbl in zip(['Base', 'Subg'],
                                                 (base_auprs, subg_auprs),
                                                 [base_lbl, subg_lbl]):
                    plot_dicts[lbl][auprs] = plt_size, mtype_lbl
                    line_dicts[lbl][auprs] = dict(c=choose_label_colour(gene))

                for ax, lbl, (base_aupr, subg_aupr) in zip(
                        [base_ax, subg_ax], ['Base', 'Subg'],
                        [base_auprs, subg_auprs]
                        ):
                    plt_max = min(1.005,
                                  max(plt_max,
                                      base_aupr + 0.11, subg_aupr + 0.11))

                    auc_bbox = (base_aupr - plt_size / 2,
                                subg_aupr - plt_size / 2, plt_size, plt_size)

                    pie_ax = inset_axes(
                        ax, width='100%', height='100%',
                        bbox_to_anchor=auc_bbox, bbox_transform=ax.transData,
                        axes_kwargs=dict(aspect='equal'), borderpad=0
                        )

                    use_clr = line_dicts[lbl][base_aupr, subg_aupr]['c']
                    pie_ax.pie(x=[best_prop, 1 - best_prop],
                               colors=[use_clr + (0.77,),
                                       use_clr + (0.29,)],
                               explode=[0.29, 0], startangle=90)

    base_ax.set_title("AUPR on all point mutations",
                      size=21, weight='semibold')
    subg_ax.set_title("AUPR on best subgrouping mutations",
                      size=21, weight='semibold')

    for ax, lbl in zip([base_ax, subg_ax], ['Base', 'Subg']):
        ax.grid(alpha=0.41, linewidth=0.9)

        ax.plot([0, plt_max], [0, 0],
                color='black', linewidth=1.5, alpha=0.89)
        ax.plot([0, 0], [0, plt_max],
                color='black', linewidth=1.5, alpha=0.89)

        ax.plot([0, plt_max], [1, 1],
                color='black', linewidth=1.5, alpha=0.89)
        ax.plot([1, 1], [0, plt_max],
                color='black', linewidth=1.5, alpha=0.89)

        ax.plot([0, plt_max], [0, plt_max],
                color='#550000', linewidth=1.7, linestyle='--', alpha=0.37)

        ax.set_xlabel("using gene-wide task's predictions",
                      size=19, weight='semibold')
        ax.set_ylabel("using best subgrouping task's predicted scores",
                      size=19, weight='semibold')

        lbl_pos = place_scatter_labels(plot_dicts[lbl], ax,
                                       plt_lims=[[plt_max / 67, plt_max]] * 2,
                                       line_dict=line_dicts[lbl])

        ax.set_xlim([-plt_max / 181, plt_max])
        ax.set_ylim([-plt_max / 181, plt_max])

    plt.savefig(
        os.path.join(plot_dir, '__'.join([args.expr_source, args.cohort]),
                     "aupr-comparisons_{}.svg".format(args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        'plot_aucs',
        description="Plots comparisons of performances of classifier tasks."
        )

    parser.add_argument('expr_source', help="a source of expression datasets")
    parser.add_argument('cohort', help="a tumour cohort")
    parser.add_argument('search_params', type=str)
    parser.add_argument('mut_lvls', type=str)
    parser.add_argument('classif', help="a mutation classifier")

    args = parser.parse_args()
    out_dir = os.path.join(base_dir,
                           '__'.join([args.expr_source, args.cohort]))

    out_files = {
        out_lbl: os.path.join(
            out_dir, "out-{}__{}__{}__{}.p.gz".format(
                out_lbl, args.search_params, args.mut_lvls, args.classif)
            )
        for out_lbl in ['pred', 'pheno', 'aucs', 'conf']
        }

    if not os.path.isfile(out_files['conf']):
        raise ValueError("No experiment output found for these parameters!")

    os.makedirs(os.path.join(plot_dir,
                             '__'.join([args.expr_source, args.cohort])),
                exist_ok=True)

    out_list = []
    for out_lbl in ['pred', 'pheno', 'aucs', 'conf']:
        with bz2.BZ2File(out_files[out_lbl], 'r') as f:
            out_list += [pickle.load(f)]

    pred_dfs, phn_dict, auc_df, conf_df = out_list
    assert auc_df.index.isin(phn_dict).all()

    plot_sub_comparisons(auc_df.Chrm, phn_dict, conf_df.Chrm, args)
    plot_aupr_comparisons(pred_dfs['Chrm'], phn_dict,
                          auc_df.Chrm, conf_df.Chrm, args)


if __name__ == '__main__':
    main()

