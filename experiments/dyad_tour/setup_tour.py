
import os
import sys
base_dir = os.path.dirname(__file__)
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.experiments.dyad_tour import *
from HetMan.experiments.subvariant_tour.setup_tour import get_cohort_data
from dryadic.features.mutations import MuType

import argparse
import synapseclient
import pandas as pd
import dill as pickle

from functools import reduce
from operator import or_
from itertools import combinations as combn
from itertools import product


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('expr_source', type=str,
                        help="which TCGA expression data source to use")

    parser.add_argument('cohort', type=str, help="which TCGA cohort to use")
    parser.add_argument(
        'samp_cutoff', type=int,
        help="minimum number of mutated samples needed to test a gene"
        )

    parser.add_argument('mut_levels', type=str,
                        help="the mutation property levels to consider")
    parser.add_argument('out_dir', type=str, default=base_dir)

    args = parser.parse_args()
    out_path = os.path.join(args.out_dir, 'setup')
    use_lvls = args.mut_levels.split('__')
    
    cdata = get_cohort_data(args.cohort, args.expr_source, use_lvls)
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
    samp_dict = {mtype: mtype.get_samples(cdata.mtree) for mtype in vars_list}
    vars_list = {mtype for mtype in vars_list
                 if (len(samp_dict[mtype])
                     <= (len(cdata.get_samples()) - args.samp_cutoff))}

    # remove mutations that are functionally equivalent to another mutation
    vars_list -= {mtype1 for mtype1, mtype2 in product(vars_list, repeat=2)
                  if (mtype1 != mtype2 and mtype1.is_supertype(mtype2)
                      and samp_dict[mtype1] == samp_dict[mtype2])}
    samp_dict = {mtype: samps for mtype, samps in samp_dict.items()
                 if mtype in vars_list}

    pairs_list = {
        tuple(sorted([mtype1, mtype2]))
        for (mtype1, samps1), (mtype2, samps2) in combn(samp_dict.items(), 2)
        if (len(samps1 - samps2) >= args.samp_cutoff
            and len(samps2 - samps1) >= args.samp_cutoff
            and (mtype1 & mtype2).is_empty())
        }

    copy_mtype = MuType({('Scale', 'Copy'): None})
    pnt_mtype = MuType({('Scale', 'Point'): None})

    if args.mut_levels != 'Exon__Location__Protein':
        pairs_list -= {[mtype1, mtype2] for mtype1, mtype2 in pairs_list
                       if ((not (mtype1 & copy_mtype).is_empty()
                            or mtype1.subtype_list()[0][1] == pnt_mtype)
                           and (not (mtype2 & copy_mtype).is_empty()
                                or mtype2.subtype_list()[0][1] == pnt_mtype))}

    with open(os.path.join(out_path, "pairs-list.p"), 'wb') as f:
        pickle.dump(sorted(pairs_list), f)
    with open(os.path.join(out_path, "pairs-count.txt"), 'w') as fl:
        fl.write(str(len(pairs_list)))


if __name__ == '__main__':
    main()

