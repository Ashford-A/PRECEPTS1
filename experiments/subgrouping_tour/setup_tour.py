
from ..utilities.mutations import pnt_mtype
from dryadic.features.mutations import MuType

from .param_list import params, mut_lvls
from ..utilities.data_dirs import vep_cache_dir, expr_sources
from ...features.data.oncoKB import get_gene_list
from ...features.cohorts.utils import get_cohort_data

import os
import argparse
import bz2
import dill as pickle
from itertools import product


def main():
    parser = argparse.ArgumentParser(
        'setup_tour',
        description="Load datasets and enumerate subgroupings to be tested."
        )

    parser.add_argument('expr_source', type=str,
                        help="a source of expression data")
    parser.add_argument('cohort', type=str, help="a tumour cohort")
    parser.add_argument('search_params', type=str,)
    parser.add_argument('mut_lvls', type=str,)
    parser.add_argument('out_dir', type=str,)

    # parse command line arguments
    args = parser.parse_args()
    out_path = os.path.join(args.out_dir, 'setup')

    lvl_lists = [('Gene', ) + lvl_list
                 for lvl_list in mut_lvls[args.mut_lvls]]
    search_dict = params[args.search_params]
    use_genes = get_gene_list(min_sources=2)

    cdata = get_cohort_data(args.cohort, args.expr_source, lvl_lists,
                            vep_cache_dir, out_path, use_genes,
                            use_copies=False)
    with bz2.BZ2File(os.path.join(out_path, "cohort-data.p.gz"), 'w') as f:
        pickle.dump(cdata, f, protocol=-1)

    total_samps = len(cdata.get_samples())
    max_samps = total_samps - search_dict['samp_cutoff']
    test_mtypes = set()

    for gene in set(dict(tuple(cdata.mtrees.values())[0])):
        pnt_count = {len(cdata.mtrees[lvls][gene].get_samples())
                     for lvls in lvl_lists}

        assert len(pnt_count) == 1, (
            "Mismatching mutation trees for gene {}!".format(gene))
        pnt_count = tuple(pnt_count)[0]

        if pnt_count >= search_dict['samp_cutoff']:
            samp_dict = {None: cdata.mtrees[lvl_lists[0]][gene].get_samples()}
            gene_types = set()

            for lvls in lvl_lists:
                use_mtree = cdata.mtrees[lvls][gene]

                lvl_types = {
                    mtype for mtype in use_mtree.combtypes(
                        comb_sizes=tuple(
                            range(1, search_dict['branch_combs'] + 1)),
                        min_type_size=search_dict['samp_cutoff'],
                        min_branch_size=search_dict['min_branch']
                        )
                    }

                samp_dict.update({mtype: mtype.get_samples(use_mtree)
                                  for mtype in lvl_types})

                gene_types |= {mtype for mtype in lvl_types
                               if (len(samp_dict[mtype]) <= max_samps
                                   and len(samp_dict[mtype]) < pnt_count)}

            rmv_mtypes = set()
            for rmv_mtype in sorted(gene_types):
                rmv_lvls = rmv_mtype.get_levels()

                for cmp_mtype in sorted(gene_types
                                        - {rmv_mtype} - rmv_mtypes):
                    cmp_lvls = cmp_mtype.get_levels()

                    if (samp_dict[rmv_mtype] == samp_dict[cmp_mtype]
                            and (rmv_mtype.is_supertype(cmp_mtype)
                                 or (any('domain' in lvl for lvl in rmv_lvls)
                                     and all('domain' not in lvl
                                             for lvl in cmp_lvls))
                                 or len(rmv_lvls) > len(cmp_lvls)
                                 or rmv_mtype > cmp_mtype)):
                        rmv_mtypes |= {rmv_mtype}
                        break

            test_mtypes |= {MuType({('Gene', gene): mtype})
                            for mtype in gene_types - rmv_mtypes}
            test_mtypes |= {MuType({('Gene', gene): None})}

    with open(os.path.join(out_path, "muts-list.p"), 'wb') as f:
        pickle.dump(sorted(test_mtypes), f, protocol=-1)
    with open(os.path.join(out_path, "muts-count.txt"), 'w') as fl:
        fl.write(str(len(test_mtypes)))


if __name__ == '__main__':
    main()

