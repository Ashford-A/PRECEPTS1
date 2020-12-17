
from .param_lists import search_params, mut_lvls
from ..utilities.data_dirs import vep_cache_dir
from ...features.data.oncoKB import get_gene_list
from ...features.cohorts.utils import get_cohort_data

from ..utilities.mutations import (pnt_mtype, shal_mtype,
                                   dup_mtype, loss_mtype, Mcomb, ExMcomb)
from dryadic.features.mutations import MuType

import os
import argparse
import bz2
import dill as pickle
import numpy as np

from itertools import combinations as combn
from itertools import product


def get_all_mtype(mtype, gene, use_mtrees, lvls_dict=None, base_lvls=None):
    _, sub_type = tuple(mtype.subtype_iter())[0]

    if sub_type in lvls_dict:
        use_lvls = lvls_dict[sub_type]
    elif base_lvls is not None:
        use_lvls = tuple(base_lvls)

    else:
        use_lvls = sorted(use_mtrees.keys())[0]

    return MuType({('Gene', gene): use_mtrees[use_lvls][gene].allkey()})


def main():
    parser = argparse.ArgumentParser(
        'setup_isolate',
        description="Load datasets and enumerate subgroupings to be tested."
        )

    parser.add_argument('expr_source', type=str,
                        help="a source of expression data")
    parser.add_argument('cohort', type=str, help="a tumour cohort")
    parser.add_argument('search_params', type=str, choices=set(search_params))
    parser.add_argument('mut_lvls', type=str, choices=set(mut_lvls))
    parser.add_argument('out_dir', type=str,
                        help="the working directory for this experiment")

    args = parser.parse_args()
    out_path = os.path.join(args.out_dir, 'setup')

    # get the combination of mutation attributes and search constraints
    # that will govern the enumeration of subgroupings
    lvl_lists = [('Gene', 'Scale', 'Copy') + lvl_list
                 for lvl_list in mut_lvls[args.mut_lvls]]
    search_dict = search_params[args.search_params]
    use_genes = get_gene_list(min_sources=4)

    # load and process the -omic datasets for this cohort
    cdata = get_cohort_data(args.cohort, args.expr_source, lvl_lists,
                            vep_cache_dir, out_path, use_genes)
    with bz2.BZ2File(os.path.join(out_path, "cohort-data.p.gz"), 'w') as f:
        pickle.dump(cdata, f, protocol=-1)

    assert sorted(cdata.mtrees.keys()) == sorted(lvl_lists), (
        "Level combination mutation trees incorrectly instantiated!")

    mut_genes = set(tuple(sorted(dict(mtree)))
                    for mtree in cdata.mtrees.values())
    assert len(mut_genes) == 1, (
        "Level combination mutation trees contain mismatching sets of genes!")

    mut_genes = tuple(mut_genes)[0]
    max_samps = len(cdata.get_samples()) - search_dict['samp_cutoff']

    test_mtypes = dict()
    test_muts = set()
    lvls_dict = dict()

    for gene in mut_genes:
        gene_mtrees = {lvls: mtree[gene]
                       for lvls, mtree in cdata.mtrees.items()}

        root_types = {
            root_type for root_type in {pnt_mtype, dup_mtype, loss_mtype,
                                        pnt_mtype | dup_mtype,
                                        pnt_mtype | loss_mtype}
            if (len(root_type.get_samples(*gene_mtrees.values()))
                >= search_dict['samp_cutoff'])
            }

        samp_dict = {mtype: mtype.get_samples(*gene_mtrees.values())
                     for mtype in root_types | {pnt_mtype}}
        pnt_types = set()

        if pnt_mtype in root_types:
            for lvls, lvl_tree in gene_mtrees.items():
                lvl_types = lvl_tree.combtypes(
                    mtype=pnt_mtype,
                    comb_sizes=tuple(
                        range(1, search_dict['branch_combs'] + 1)),
                    min_type_size=search_dict['samp_cutoff'],
                    min_branch_size=search_dict['min_branch']
                    ) - {pnt_mtype}

                samp_dict.update({mtype: mtype.get_samples(lvl_tree)
                                  for mtype in lvl_types})
                lvls_dict.update({mtype: lvls for mtype in lvl_types})
                pnt_types |= lvl_types

        gene_types = {mtype for mtype in pnt_types | root_types
                      if samp_dict[mtype] != samp_dict[pnt_mtype]}
        lf_dict = {mtype: mtype.leaves() for mtype in gene_types}

        test_list = list(gene_types - root_types)
        samp_list = np.array([samp_dict[mtype] for mtype in test_list])
        eq_indx = np.where(np.triu(np.equal.outer(samp_list, samp_list), k=1))
        dup_indx = set()

        for indx1, indx2 in zip(*eq_indx):
            for i, j in [[indx1, indx2], [indx2, indx1]]:
                if (test_list[i].is_supertype(test_list[j])
                        or (lvl_lists.index(lvls_dict[test_list[i]])
                            == lvl_lists.index(lvls_dict[test_list[i]])
                            and (len(lf_dict[test_list[i]])
                                 > len(lf_dict[test_list[j]])))
                        or (lvl_lists.index(lvls_dict[test_list[i]])
                            > lvl_lists.index(lvls_dict[test_list[j]]))):
                    dup_indx |= {i}

        rmv_mtypes = {test_list[i] for i in dup_indx}
        gene_mtypes = {MuType({('Gene', gene): mtype})
                       for mtype in gene_types - rmv_mtypes | {pnt_mtype}
                       if (search_dict['samp_cutoff']
                           <= len(samp_dict[mtype]) <= max_samps)}

        test_muts |= gene_mtypes
        if gene_mtypes:
            test_mtypes[gene] = gene_mtypes

    for (gene1, mtypes1), (gene2, mtypes2) in combn(test_mtypes.items(), 2):
        pair_combs = set()

        ex_mtypes = [MuType({}), MuType({
            ('Gene', (gene1, gene2)): shal_mtype})]

        for mtype in mtypes1 | mtypes2:
            all_mtype = get_all_mtype(mtype, gene1, cdata.mtrees, lvls_dict,
                                      lvl_lists[0])
            all_mtype |= get_all_mtype(mtype, gene2, cdata.mtrees, lvls_dict,
                                       lvl_lists[0])
            pair_combs |= {ExMcomb(all_mtype - ex_mtype, mtype)
                           for ex_mtype in ex_mtypes}

        #TODO: consider pairs within genes and pairs between genes separately?
        for mtype1, mtype2 in product(mtypes1, mtypes2):
            pair_combs |= {Mcomb(mtype1, mtype2)}

            all_mtype = get_all_mtype(mtype1, gene1, cdata.mtrees, lvls_dict,
                                      lvl_lists[0])
            all_mtype |= get_all_mtype(mtype2, gene2, cdata.mtrees, lvls_dict,
                                       lvl_lists[0])
            pair_combs |= {ExMcomb(all_mtype - ex_mtype, mtype1, mtype2)
                           for ex_mtype in ex_mtypes}

        test_muts |= {mcomb for mcomb in pair_combs
                      if (search_dict['samp_cutoff']
                          <= len(mcomb.get_samples(*cdata.mtrees.values()))
                          <= max_samps)}

    with open(os.path.join(out_path, "muts-list.p"), 'wb') as f:
        pickle.dump(sorted(test_muts), f, protocol=-1)
    with open(os.path.join(out_path, "muts-count.txt"), 'w') as fl:
        fl.write(str(len(test_muts)))


if __name__ == '__main__':
    main()

