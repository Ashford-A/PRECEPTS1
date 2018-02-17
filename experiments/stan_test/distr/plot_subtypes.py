
import os
base_dir = os.path.dirname(__file__)
plot_dir = os.path.join(base_dir, 'plots', 'subtypes')

import sys
sys.path.extend([os.path.join(base_dir, '../../../..')])

from HetMan.features.variants import MuType
from HetMan.features.cohorts.mut import VariantCohort

import argparse
import synapseclient
import dill as pickle
import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns

firehose_dir = '/home/exacloud/lustre1/share_your_data_here/precepts/firehose'


def load_infer_mats(model_name, cohort, gene):

    out_lists = [
        pickle.load(open(os.path.join(
            base_dir, "output", model_name, cohort, gene,
            "out__cv-{}.p".format(cv_id)
            ), 'rb'))
        for cv_id in range(10)
        ]

    return np.concatenate(out_lists, axis=1)


def plot_label_stability(out_data, args, cdata):
    fig, ax = plt.subplots(figsize=(13, 8))
    kern_bw = (np.max(out_data) - np.min(out_data)) / 40
    mut_clr = sns.hls_palette(1, l=.4, s=.9)[0]

    use_mtype = MuType({('Gene', args.gene): None})
    mtype_stat = np.array(cdata.train_pheno(use_mtype))
    out_meds = np.percentile(out_data, q=50, axis=1)

    ax = sns.kdeplot(out_meds[~mtype_stat], color='0.4', alpha=0.6,
                     shade=False, linewidth=2.4, bw=kern_bw, gridsize=1000,
                     label='Wild-Type')
    ax = sns.kdeplot(out_meds[mtype_stat], color=mut_clr, alpha=0.6,
                     shade=False, linewidth=2.4, bw=kern_bw, gridsize=1000,
                     label='{} Mutant'.format(args.gene))

    for i in range(out_data.shape[1]):
        ax = sns.kdeplot(out_data[~mtype_stat, i],
                         shade=True, alpha=0.05, linewidth=0, color='0.4',
                         bw=kern_bw, gridsize=1000)
        ax = sns.kdeplot(out_data[mtype_stat, i],
                         shade=True, alpha=0.05, linewidth=0, color=mut_clr,
                         bw=kern_bw, gridsize=1000)

    plt.xlabel('Inferred Mutation Score', fontsize=20)
    plt.ylabel('Density', fontsize=20)

    fig.savefig(os.path.join(plot_dir,
                             'label_stability__{}-{}-{}.png'.format(
                                 args.model_name, args.cohort, args.gene)),
                dpi=200, bbox_inches='tight')
    plt.close()


def main():
    """Runs the experiment."""

    parser = argparse.ArgumentParser(
        description=("Find the signatures a classifier predicts for a list "
                     "of sub-types.")
        )

    parser.add_argument('model_name', type=str, help='label of a model')
    parser.add_argument('cohort', type=str, help='a TCGA cohort')
    parser.add_argument('gene', type=str, help='a gene')

    args = parser.parse_args()
    os.makedirs(plot_dir, exist_ok=True)
    infer_mat = load_infer_mats(args.model_name, args.cohort, args.gene)

    # logs into Synapse using locally-stored credentials
    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = ('/home/exacloud/lustre1/share_your_data_here'
                                '/precepts/synapse')
    syn.login()

    cdata = VariantCohort(
        cohort=args.cohort, mut_genes=[args.gene],
        mut_levels=['Gene', 'Form_base', 'Exon'], expr_source='Firehose',
        data_dir=firehose_dir, syn=syn, cv_prop=1.0
        )

    plot_label_stability(infer_mat, args, cdata)


if __name__ == "__main__":
    main()

