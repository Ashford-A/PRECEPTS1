
import os
import sys
base_dir = os.path.dirname(__file__)
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.experiments.variant_baseline import *
from HetMan.experiments.utilities.load_input import parse_subtypes
from HetMan.features.cohorts.tcga import MutationCohort
from HetMan.features.cohorts.beatAML import BeatAmlCohort
from dryadic.features.mutations import MuType

import argparse
import synapseclient
import pandas as pd
import dill as pickle

from functools import reduce
from operator import or_
from itertools import product


def get_cohort_data(cohort, expr_source=None, cv_seed=None, test_prop=0):
    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = syn_root
    syn.login()

    gene_df = pd.read_csv(gene_list, sep='\t', skiprows=1, index_col=0)
    use_genes = gene_df.index[
        (gene_df.loc[
            :, ['Vogelstein', 'SANGER CGC(05/30/2017)',
                'FOUNDATION ONE', 'MSK-IMPACT']]
            == 'Yes').sum(axis=1) > 1
        ]

    if cohort == 'beatAML':
        cdata = BeatAmlCohort(mut_genes=use_genes.tolist(),
                              mut_levels=['Gene', 'Form', 'Exon', 'Protein'],
                              expr_file=beatAML_files['expr'],
                              samp_file=beatAML_files['samps'], syn=syn,
                              annot_file=annot_file, cv_seed=cv_seed,
                              test_prop=test_prop)

    else:
        if expr_source is None or expr_source == 'default':
            raise ValueError("If using a TCGA cohort, you must provide a "
                             "source of expression data!")

        source_info = expr_source.split('__')
        source_base = source_info[0]
        collapse_txs = not (len(source_info) > 1 and source_info[1] == 'txs')

        cdata = MutationCohort(
            cohort=cohort.split('_')[0],
            mut_levels=['Gene', 'Form_base', 'Protein'],
            mut_genes=use_genes.tolist(), expr_source=source_base,
            var_source='mc3', copy_source='Firehose', annot_file=annot_file,
            type_file=type_file, expr_dir=expr_sources[source_base],
            copy_dir=copy_dir, collapse_txs=collapse_txs, syn=syn,
            cv_seed=cv_seed, test_prop=test_prop, annot_fields=['transcript'],
            use_types=parse_subtypes(cohort)
            )

    return cdata


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('expr_source', type=str,
                        help="which TCGA expression data source to use")

    parser.add_argument('cohort', type=str, help="which TCGA cohort to use")
    parser.add_argument(
        'samp_cutoff', type=int,
        help="minimum number of affected samples needed to test a mutation"
        )

    parser.add_argument('--setup_dir', type=str, default=base_dir)
    args = parser.parse_args()
    out_path = os.path.join(args.setup_dir, 'setup')
    cdata = get_cohort_data(args.cohort, args.expr_source)

    with open(os.path.join(out_path, "cohort-data.p"), 'wb') as f:
        pickle.dump(cdata, f)

    # find subsets of point mutations with enough affected samples for each
    # mutated gene in the cohort
    vars_list = reduce(
        or_,
        [{MuType({('Gene', gene): mtype})
          for mtype in muts['Point'].branchtypes(min_size=args.samp_cutoff)}
         for gene, muts in cdata.mtree
         if ('Scale', 'Point') in muts.allkey()], set()
        )

    # add copy number deletions for each gene if enough samples are affected
    vars_list |= {MuType({('Gene', gene): {('Copy', 'DeepDel'): None}})
                  for gene, muts in cdata.mtree
                  if (('Scale', 'Copy') in muts.allkey()
                      and ('Copy', 'DeepDel') in muts['Copy'].allkey()
                      and len(muts['Copy']['DeepDel']) >= args.samp_cutoff)}

    # add copy number amplifications for each gene
    vars_list |= {MuType({('Gene', gene): {('Copy', 'DeepGain'): None}})
                  for gene, muts in cdata.mtree
                  if (('Scale', 'Copy') in muts.allkey()
                      and ('Copy', 'DeepGain') in muts['Copy'].allkey()
                      and len(muts['Copy']['DeepGain']) >= args.samp_cutoff)}

    # add all point mutations as a single mutation type for each gene if it
    # contains more than one type of point mutation
    vars_list |= {MuType({('Gene', gene): {('Scale', 'Point'): None}})
                  for gene, muts in cdata.mtree
                  if (('Scale', 'Point') in muts.allkey()
                      and len(muts['Point'].allkey()) > 1
                      and len(muts['Point']) >= args.samp_cutoff)}

    # filter out mutations that do not have enough wild-type samples
    vars_list = {mtype for mtype in vars_list
                 if (len(mtype.get_samples(cdata.mtree))
                     <= (len(cdata.get_samples()) - args.samp_cutoff))}

    # remove mutations that are functionally equivalent to another mutation
    vars_list -= {mtype1 for mtype1, mtype2 in product(vars_list, repeat=2)
                  if (mtype1 != mtype2 and mtype1.is_supertype(mtype2)
                      and (mtype1.get_samples(cdata.mtree)
                           == mtype2.get_samples(cdata.mtree)))}

    # save the enumerated mutations, and the number of such mutations, to file
    with open(os.path.join(out_path, "vars-list.p"), 'wb') as fl:
        pickle.dump(sorted(vars_list), fl)
    with open(os.path.join(out_path, "vars-count.txt"), 'w') as f:
        f.write(str(len(vars_list)))


if __name__ == '__main__':
    main()

