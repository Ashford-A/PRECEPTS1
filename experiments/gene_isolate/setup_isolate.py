
from .param_lists import search_params, mut_lvls
from ..utilities.data_dirs import choose_source, vep_cache_dir
from ...features.cohorts.utils import get_cohort_data

from ..utilities.mutations import (pnt_mtype, shal_mtype, dup_mtype,
                                   gains_mtype, loss_mtype, dels_mtype,
                                   Mcomb, ExMcomb)
from dryadic.features.mutations import MuType

import os
import argparse
import bz2
import dill as pickle

import numpy as np
from itertools import combinations as combn
from functools import reduce
from operator import or_


def main():
    parser = argparse.ArgumentParser(
        'setup_isolate',
        description="Load datasets and enumerate subgroupings to be tested."
        )

    parser.add_argument('gene', type=str, help="a mutated gene")
    parser.add_argument('cohort', type=str, help="a tumour cohort")
    parser.add_argument('mut_lvls', type=str,
                        help="a set of mutation attribute hierarchies")
    parser.add_argument('search_params', type=str)
    parser.add_argument('out_dir', type=str)

    args = parser.parse_args()
    out_path = os.path.join(args.out_dir, 'setup')

    # get the combinations of mutation attributes and search constraints
    # that will govern the enumeration of subgroupings within this gene
    lvl_lists = [('Scale', 'Copy') + lvl_list
                 for lvl_list in mut_lvls[args.mut_lvls]]
    search_dict = search_params[args.search_params]

    cdata = get_cohort_data(args.cohort, choose_source(args.cohort),
                            lvl_lists, vep_cache_dir, out_path, {args.gene})
    with bz2.BZ2File(os.path.join(out_path, "cohort-data.p.gz"), 'w') as f:
        pickle.dump(cdata, f, protocol=-1)

    assert sorted(cdata.mtrees.keys()) == sorted(lvl_lists), (
        "Level combination mutation trees incorrectly instantiated!")

    total_samps = len(cdata.get_samples())
    max_samps = total_samps - search_dict['samp_cutoff']
    base_mtree = tuple(cdata.mtrees.values())[0]
    copy_types = set()

    if 'Copy' in dict(base_mtree):
        if 'DeepGain' in dict(base_mtree['Copy']):
            copy_types |= {dup_mtype}
        if 'DeepDel' in dict(base_mtree['Copy']):
            copy_types |= {loss_mtype}
        if 'ShalGain' in dict(base_mtree['Copy']):
            copy_types |= {gains_mtype}
        if 'ShalDel' in dict(base_mtree['Copy']):
            copy_types |= {dels_mtype}

    samp_dict = {mtype: mtype.get_samples(base_mtree) for mtype in copy_types}
    copy_types = {mtype for mtype in copy_types
                  if samp_dict[mtype] - base_mtree['Point'].get_samples()}

    root_types = {pnt_mtype} | copy_types
    root_types |= {pnt_mtype | copy_type for copy_type in copy_types}
    samp_dict.update({mtype: mtype.get_samples(base_mtree)
                      for mtype in root_types - copy_types})
    lvl_types = dict()

    for lvls in lvl_lists:
        test_types = cdata.mtrees[lvls].combtypes(
            mtype=pnt_mtype,
            comb_sizes=tuple(range(1, search_dict['branch_combs'] + 1)),
            min_type_size=search_dict['samp_cutoff'],
            min_branch_size=search_dict['min_branch']
            ) - {pnt_mtype}

        samp_dict.update({mtype: mtype.get_samples(cdata.mtrees[lvls])
                          for mtype in test_types})

        lvl_types[lvls] = {
            mtype for mtype in test_types
            if ((search_dict['samp_cutoff']
                 <= len(samp_dict[mtype]) <= max_samps)
                and samp_dict[mtype] != samp_dict[pnt_mtype])
            }

    test_mtypes = reduce(or_, lvl_types.values())
    lvl_dict = {mtype: mtype.get_levels() for mtype in test_mtypes}
    lf_dict = {mtype: mtype.leaves() for mtype in test_mtypes}

    test_list = list(test_mtypes)
    samp_list = np.array([samp_dict[mtype] for mtype in test_list])
    eq_indx = np.where(np.triu(np.equal.outer(samp_list, samp_list), k=1))
    dup_indx = set()

    for indx1, indx2 in zip(*eq_indx):
        for i, j in [[indx1, indx2], [indx2, indx1]]:
            if (test_list[i].is_supertype(test_list[j])
                    or (len(lf_dict[test_list[i]])
                        > len(lf_dict[test_list[j]]))
                    or (len(lvl_dict[test_list[i]])
                        > len(lvl_dict[test_list[j]]))):
                dup_indx |= {i}

    rmv_mtypes = {test_list[i] for i in dup_indx}
    test_mcombs = set()
    mtype_lvls = {mtype: mtype.get_levels() - {'Scale'}
                  for mtype in (test_mtypes | root_types) - rmv_mtypes}

    for lvls in lvl_lists:
        all_mtype = MuType(cdata.mtrees[lvls].allkey())
        test_combs = set()

        test_types = lvl_types[lvls] - rmv_mtypes
        if lvls == sorted(lvl_lists)[0]:
            test_types |= root_types

        lvl_pairs = {
            (mtype2, mtype1) if 'Copy' in mtype_lvls[mtype1]
            else (mtype1, mtype2)
            for mtype1, mtype2 in combn(sorted(test_types | root_types), 2)
            if (('Copy' not in mtype_lvls[mtype1]
                 or 'Copy' not in mtype_lvls[mtype2])
                and (mtype1 & mtype2).is_empty()
                and not samp_dict[mtype1] >= samp_dict[mtype2]
                and not samp_dict[mtype2] >= samp_dict[mtype1])
            }

        for mtype in test_types:
            excomb = ExMcomb(all_mtype, mtype)
            samp_dict[excomb] = excomb.get_samples(cdata.mtrees[lvls])

            test_combs |= {excomb}
            if samp_dict[excomb] == samp_dict[mtype]:
                test_mtypes -= {mtype}

            excomb_shal = ExMcomb(all_mtype - shal_mtype, mtype)
            exshal_samps = excomb_shal.get_samples(cdata.mtrees[lvls])
            if exshal_samps != samp_dict[excomb]:
                samp_dict[excomb_shal] = exshal_samps
                test_combs |= {excomb_shal}

        for mtype1, mtype2 in sorted(lvl_pairs):
            pair_comb = Mcomb(mtype1, mtype2)
            comb_samps = pair_comb.get_samples(cdata.mtrees[lvls])
            expair = ExMcomb(all_mtype, mtype1, mtype2)
            expair_samps = expair.get_samples(cdata.mtrees[lvls])

            base_samps1 = ExMcomb(
                all_mtype, mtype1).get_samples(cdata.mtrees[lvls])
            base_samps2 = ExMcomb(
                all_mtype, mtype2).get_samples(cdata.mtrees[lvls])

            if comb_samps != base_samps1 or comb_samps != base_samps2:
                if comb_samps != expair_samps:
                    samp_dict[pair_comb] = comb_samps
                    test_combs |= {pair_comb}

                if comb_samps == samp_dict[mtype1]:
                    test_mtypes -= {mtype1}
                if comb_samps == samp_dict[mtype2]:
                    test_mtypes -= {mtype2}

            if ((expair_samps != base_samps1 or expair_samps != base_samps2)
                and not any(expair_samps == smps
                            for mcb, smps in samp_dict.items()
                            if (isinstance(mcb, ExMcomb)
                                and len(mcb.mtypes) == 2))):
                samp_dict[expair] = expair_samps
                test_combs |= {expair}

            bshl_samps1 = ExMcomb(
                all_mtype - shal_mtype, mtype1).get_samples(
                cdata.mtrees[lvls])
            bshl_samps2 = ExMcomb(
                all_mtype - shal_mtype, mtype2).get_samples(
                cdata.mtrees[lvls])

            expair_shal = ExMcomb(all_mtype - shal_mtype, mtype1, mtype2)
            exshal_samps = expair_shal.get_samples(cdata.mtrees[lvls])

            if (exshal_samps != expair_samps
                    and (exshal_samps != bshl_samps1
                         or exshal_samps != bshl_samps2)
                    and not any(exshal_samps == smps
                                for mcb, smps in samp_dict.items()
                                if (isinstance(mcb, ExMcomb)
                                    and len(mcb.mtypes) == 2))):
                samp_dict[expair_shal] = exshal_samps
                test_combs |= {expair_shal}

        test_mcombs |= {mcomb for mcomb in test_combs
                        if (search_dict['samp_cutoff']
                            <= len(samp_dict[mcomb]) <= max_samps)}

    test_muts = {mut for mut in ((test_mtypes - rmv_mtypes)
                                 | test_mcombs | root_types)
                 if (search_dict['samp_cutoff']
                     <= len(samp_dict[mut]) <= max_samps)}

    with open(os.path.join(out_path, "muts-list.p"), 'wb') as f:
        pickle.dump(sorted(test_muts), f, protocol=-1)
    with open(os.path.join(out_path, "muts-count.txt"), 'w') as fl:
        fl.write(str(len(test_muts)))


if __name__ == '__main__':
    main()

