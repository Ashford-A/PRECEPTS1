
from .param_list import params
from ..utilities.data_dirs import vep_cache_dir
from ...features.data.oncoKB import get_gene_list
from ...features.cohorts.utils import get_cohort_data

from ..utilities.mutations import (pnt_mtype, copy_mtype, shal_mtype,
                                   dup_mtype, gains_mtype, loss_mtype,
                                   dels_mtype, Mcomb, ExMcomb)
from dryadic.features.mutations import MuType

import os
import argparse
import bz2
import dill as pickle
from itertools import combinations as combn


def main():
    parser = argparse.ArgumentParser(
        'setup_isolate',
        description="Load datasets and enumerate subgroupings to be tested."
        )

    parser.add_argument('expr_source', type=str,
                        help="a source of expression data")
    parser.add_argument('cohort', type=str, help="a tumour cohort")
    parser.add_argument('mut_levels', type=str,
                        help="a combination of mutation attribute levels")
    parser.add_argument('search_params', type=str, choices=set(params))
    parser.add_argument('out_dir', type=str,
                        help="the working directory for this experiment")

    # parse command line arguments, figure out where output will be stored
    args = parser.parse_args()
    out_path = os.path.join(args.out_dir, 'setup')

    # get the combination of mutation attributes and search constraints
    # that will govern the enumeration of subgroupings
    lvl_list = ('Gene', 'Scale', 'Copy') + tuple(args.mut_levels.split('__'))
    search_dict = params[args.search_params]
    use_genes = get_gene_list(min_sources=2)

    # load and process the -omic datasets for this cohort
    cdata = get_cohort_data(args.cohort, args.expr_source, lvl_list,
                            vep_cache_dir, out_path, use_genes)
    with bz2.BZ2File(os.path.join(out_path, "cohort-data.p.gz"), 'w') as f:
        pickle.dump(cdata, f, protocol=-1)

    # get the maximum number of samples allowed per subgrouping, initialize
    # the list of enumerated subgroupings
    max_samps = len(cdata.get_samples()) - search_dict['samp_cutoff']
    test_muts = set()

    # for each gene with enough point mutations, find all of the combinations
    # of its point mutation subtypes that satisfy the search criteria
    for gene, mtree in cdata.mtrees[lvl_list]:
        if len(pnt_mtype.get_samples(mtree)) >= search_dict['samp_cutoff']:
            comb_types = mtree.combtypes(
                mtype=pnt_mtype,
                comb_sizes=tuple(range(1, search_dict['branch_combs'] + 1)),
                min_type_size=search_dict['samp_cutoff'],
                min_branch_size=search_dict['min_branch']
                ) - {pnt_mtype}

            # get the samples mutated for each subtype combination in this
            # cohort; remove subtypes that span all of the gene's mutations
            samp_dict = {mtype: mtype.get_samples(mtree)
                         for mtype in comb_types}
            samp_dict[pnt_mtype] = mtree['Point'].get_samples()
            pnt_types = {mtype for mtype in comb_types
                         if samp_dict[mtype] != samp_dict[pnt_mtype]}

            # remove subtypes that are mutated in the same set of samples as
            # another subtype and are less granular in their definition
            rmv_mtypes = set()
            for rmv_mtype in sorted(pnt_types):
                rmv_lvls = rmv_mtype.get_levels()

                # e.g. remove `Missense` in favour of `Missense->5th Exon` if
                # all of this gene's missense mutations are on the fifth exon
                for cmp_mtype in pnt_types - {rmv_mtype} - rmv_mtypes:
                    if (samp_dict[rmv_mtype] == samp_dict[cmp_mtype]
                            and (rmv_mtype.is_supertype(cmp_mtype)
                                 or (len(rmv_lvls)
                                     < len(cmp_mtype.get_levels())))):
                        rmv_mtypes |= {rmv_mtype}
                        break

            # only add the gene-wide point mutation subtype if we are using
            # the "base" combination of mutation attributes
            pnt_types -= rmv_mtypes
            if args.mut_levels == 'Consequence__Exon':
                pnt_types |= {pnt_mtype}

            # add subgroupings composed solely of CNAs where applicable
            if 'Copy' in dict(mtree):
                copy_types = {dup_mtype, loss_mtype}

                if 'ShalGain' in dict(mtree['Copy']):
                    copy_types |= {gains_mtype}
                if 'ShalDel' in dict(mtree['Copy']):
                    copy_types |= {dels_mtype}

            else:
                copy_types = set()

            # find the enumerated point mutations for this gene that can be
            # combined with CNAs to produce a novel set of mutated samples
            copy_dyads = set()
            for copy_type in copy_types:
                samp_dict[copy_type] = copy_type.get_samples(mtree)

                if len(samp_dict[copy_type]) >= 5:
                    for pnt_type in pnt_types:
                        new_dyad = pnt_type | copy_type
                        dyad_samps = new_dyad.get_samples(mtree)

                        if (dyad_samps > samp_dict[pnt_type]
                                and dyad_samps > samp_dict[copy_type]):
                            copy_dyads |= {new_dyad}
                            samp_dict[new_dyad] = dyad_samps

            # add the CNA-only subgroupings if we are using "base" attributes
            test_types = pnt_types | copy_dyads
            if args.mut_levels == 'Consequence__Exon':
                test_types |= copy_types

            # check that the gene's enumerated subgroupings satisfy recurrence
            # thresholds before adding them to the final list
            test_muts |= {MuType({('Gene', gene): mtype})
                          for mtype in test_types
                          if (search_dict['samp_cutoff']
                              <= len(samp_dict[mtype]) <= max_samps)}

            all_mtype = MuType(mtree.allkey())
            ex_mtypes = [MuType({}), shal_mtype]
            mtype_lvls = {mtype: mtype.get_levels() - {'Scale'}
                          for mtype in pnt_types | copy_types}

            use_pairs = {(mtype2, mtype1)
                         if 'Copy' in lvls1 else (mtype1, mtype2)
                         for (mtype1, lvls1), (mtype2, lvls2)
                         in combn(mtype_lvls.items(), 2)
                         if (('Copy' not in lvls1 or 'Copy' not in lvls2)
                             and (mtype1 & mtype2).is_empty()
                             and not samp_dict[mtype1] >= samp_dict[mtype2]
                             and not samp_dict[mtype2] >= samp_dict[mtype1])}

            use_mcombs = {Mcomb(*pair) for pair in use_pairs}
            use_mcombs |= {ExMcomb(all_mtype - ex_mtype, *pair)
                           for pair in use_pairs for ex_mtype in ex_mtypes}

            use_mcombs |= {ExMcomb(all_mtype - ex_mtype, mtype)
                           for mtype in test_types
                           for ex_mtype in ex_mtypes}

            test_muts |= {
                Mcomb(*[MuType({('Gene', gene): mtype})
                        for mtype in mcomb.mtypes])
                if isinstance(mcomb, Mcomb)

                else ExMcomb(MuType({('Gene', gene): mcomb.all_mtype}),
                             *[MuType({('Gene', gene): mtype})
                               for mtype in mcomb.mtypes])

                for mcomb in use_mcombs
                if (isinstance(mcomb, (Mcomb, ExMcomb))
                    and (search_dict['samp_cutoff']
                         <= len(mcomb.get_samples(mtree)) <= max_samps))
                }

    # save enumerated subgroupings and number of subgroupings to file
    with open(os.path.join(out_path, "muts-list.p"), 'wb') as f:
        pickle.dump(sorted(test_muts), f, protocol=-1)
    with open(os.path.join(out_path, "muts-count.txt"), 'w') as fl:
        fl.write(str(len(test_muts)))


if __name__ == '__main__':
    main()

