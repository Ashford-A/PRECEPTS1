
import os
import sys
base_dir = os.path.dirname(__file__)
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.experiments.dyad_infer import *
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
from itertools import combinations as combn
from itertools import product


def get_cohort_data(cohort):
    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = syn_root
    syn.login()

    gene_df = pd.read_csv(gene_list, sep='\t', skiprows=1, index_col=0)
    use_genes = gene_df.index[
        (gene_df.loc[
            :, ['Vogelstein', 'SANGER CGC(05/30/2017)',
                'FOUNDATION ONE', 'MSK-IMPACT']]
            == 'Yes').sum(axis=1) >= 1
        ]

    if cohort == 'beatAML':
        cdata = BeatAmlCohort(mut_levels=['Gene', 'Form_base', 'Form',
                                          'Exon', 'Location', 'Protein'],
                              mut_genes=use_genes.tolist(),
                              expr_source='toil__gns',
                              expr_file=beatAML_files['expr'],
                              samp_file=beatAML_files['samps'], syn=syn,
                              annot_file=annot_file, cv_seed=671, test_prop=0)

    else:
        cdata = MutationCohort(
            cohort=cohort.split('_')[0],
            mut_levels=['Gene', 'Form_base', 'Form',
                        'Exon', 'Location', 'Protein'],
            mut_genes=use_genes.tolist(), expr_source='Firehose',
            var_source='mc3', copy_source='Firehose', annot_file=annot_file,
            type_file=type_file, expr_dir=expr_dir, copy_dir=copy_dir,
            syn=syn, cv_seed=671, test_prop=0, annot_fields=['transcript'],
            use_types=parse_subtypes(cohort)
            )

    return cdata


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('cohort', type=str, help="which TCGA cohort to use")
    parser.add_argument(
        'samp_cutoff', type=int,
        help="minimum number of mutated samples needed to test a gene"
        )

    parser.add_argument('--setup_dir', type=str, default=base_dir)
    args = parser.parse_args()
    out_path = os.path.join(args.setup_dir, 'setup')
    cdata = get_cohort_data(args.cohort)

    with open(os.path.join(out_path, "cohort-data.p"), 'wb') as cdata_fl:
        pickle.dump(cdata, cdata_fl)

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

    # get the samples carrying each remaining mutation
    samp_dict = {mtype: mtype.get_samples(cdata.mtree) for mtype in vars_list}

    pairs_list = {
        tuple(sorted([mtype1, mtype2]))
        for (mtype1, samps1), (mtype2, samps2) in combn(samp_dict.items(), 2)
        if (len(samps1 - samps2) >= args.samp_cutoff
            and len(samps2 - samps1) >= args.samp_cutoff
            and (mtype1 & mtype2).is_empty())
        }

    with open(os.path.join(out_path, "pairs-list.p"), 'wb') as f:
        pickle.dump(sorted(pairs_list), f)
    with open(os.path.join(out_path, "pairs-count.txt"), 'w') as fl:
        fl.write(str(len(pairs_list)))


if __name__ == '__main__':
    main()

