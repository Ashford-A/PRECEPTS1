
import os
base_dir = os.path.dirname(__file__)
plot_dir = os.path.join(base_dir, 'plots', 'subtypes')

import sys
sys.path.extend([os.path.join(base_dir, '../../../..')])

from HetMan.features.cohorts.tcga import MutationCohort
from HetMan.features.mutations import MuType
from HetMan.experiments.stan_test.distr.fit_models import load_output

import numpy as np
import pandas as pd

import argparse
import synapseclient

from itertools import chain, combinations
from functools import reduce
from operator import or_

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

firehose_dir = '/home/exacloud/lustre1/share_your_data_here/precepts/firehose'



def plot_subtype_violins(out_data, args, cdata, use_levels):
    fig, ax = plt.subplots(figsize=(9, 12))

    use_mtype = MuType({('Gene', args.gene): None})
    use_types = cdata.train_mut.branchtypes(sub_levels=use_levels)
    all_type = reduce(or_, use_types)

    out_meds = np.percentile(out_data, q=50, axis=1)
    mtype_meds = [('Wild-Type',
                   out_meds[~np.array(cdata.train_pheno(all_type))])]

    mtype_meds += [(str(mtype),
                    out_meds[np.array(cdata.train_pheno(mtype))])
                   for mtype in use_types]
    mtype_meds = sorted(mtype_meds, key=lambda x: np.mean(x[1]))

    med_df = pd.concat(pd.DataFrame({'Subtype': mtype, 'Score': meds})
                       for mtype, meds in mtype_meds)

    ax = sns.violinplot(data=med_df, x='Subtype', y='Score')

    fig.savefig(
        os.path.join(plot_dir,
                     'violins__{}-{}_{}-{}__levels_{}.png'.format(
                         args.model_name, args.solve_method,
                         args.cohort, args.gene, '__'.join(use_levels)
                        )),
        dpi=250, bbox_inches='tight'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        "Plot the distribution of labels by mutation subtype returned by a "
        "Stan classifier trained to predict all the mutations for a given "
        "gene in a TCGA cohort."
        )

    parser.add_argument('model_name', type=str, help="label of a Stan model")
    parser.add_argument('solve_method', type=str,
                        help=("method used to obtain estimates for the "
                              "parameters of the model"))

    parser.add_argument('cohort', type=str, help="a TCGA cohort")
    parser.add_argument('gene', type=str, help="a mutated gene")

    parser.add_argument('mut_levels', nargs='*',
                        default=['Form_base', 'Exon'],
                        help="which mutation annotation levels to consider")

    args = parser.parse_args()
    os.makedirs(plot_dir, exist_ok=True)
    infer_mat = load_output(args.model_name, args.solve_method,
                            args.cohort, args.gene)

    # log into Synapse using locally stored credentials
    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = ('/home/exacloud/lustre1/CompBio'
                                '/mgrzad/input-data/synapse')
    syn.login()

    cdata = MutationCohort(
        cohort=args.cohort, mut_genes=[args.gene], mut_levels=args.mut_levels,
        expr_source='Firehose', expr_dir=firehose_dir, var_source='mc3',
        syn=syn, cv_prop=1.0
        )

    for use_levels in chain.from_iterable(combinations(args.mut_levels, r)
                                          for r in range(
                                              1, len(args.mut_levels) + 1)):
        plot_subtype_violins(infer_mat, args, cdata, use_levels)


if __name__ == "__main__":
    main()

