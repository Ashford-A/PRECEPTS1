
from .param_lists import search_params, mut_lvls
from ..utilities.data_dirs import choose_source, vep_cache_dir
from ..subgrouping_isolate.setup_isolate import get_input_datasets
from dryadic.features.data.vep import process_variants

from ..utilities.mutations import (
    pnt_mtype, shal_mtype, dup_mtype, gains_mtype, loss_mtype, dels_mtype,
    Mcomb, ExMcomb
    )
from dryadic.features.mutations import MuType
from ..subgrouping_isolate.utils import IsoMutationCohort

import os
import argparse
import bz2
import dill as pickle

import pandas as pd
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

    data_dict = get_input_datasets(
        args.cohort, choose_source(args.cohort), use_genes=[args.gene],
        mut_fields=['Sample', 'Gene', 'Chr', 'Start', 'End',
                    'Strand', 'RefAllele', 'TumorAllele']
        )

    var_df = pd.DataFrame({'Chr': data_dict['vars'].Chr.astype('int'),
                           'Start': data_dict['vars'].Start.astype('int'),
                           'End': data_dict['vars'].End.astype('int'),
                           'RefAllele': data_dict['vars'].RefAllele,
                           'VarAllele': data_dict['vars'].TumorAllele,
                           'Strand': data_dict['vars'].Strand,
                           'Sample': data_dict['vars'].Sample})

    lvl_lists = [('Scale', 'Copy') + lvl_list
                 for lvl_list in mut_lvls[args.mut_lvls]]
    search_dict = search_params[args.search_params]

    var_fields = {'Gene', 'Canonical', 'Location', 'VarAllele'}
    for lvl_list in lvl_lists:
        for lvl in lvl_list[2:]:
            if '-domain' in lvl and 'Domains' not in var_fields:
                var_fields |= {'Domains'}
            else:
                var_fields |= {lvl}

    variants = process_variants(
        var_df, out_fields=var_fields, cache_dir=vep_cache_dir,
        temp_dir=out_path, assembly=data_dict['assembly'],
        distance=0, consequence_choose='pick', forks=4, update_cache=False
        )

    variants = variants.loc[(variants.CANONICAL == 'YES')
                            & variants.Gene.isin(data_dict['use_genes'])]
    copies = data_dict['copy'].loc[
        data_dict['copy'].Gene.isin(data_dict['use_genes'])]

    assert not variants.duplicated().any(), (
        "Variant data contains {} duplicate entries!".format(
            variants.duplicated().sum())
        )

    cdata = IsoMutationCohort(data_dict['expr'], variants, lvl_lists, copies,
                              data_dict['annot'], leaf_annot=None)
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

        copy_dyads = set()
        for copy_type in copy_types:
            for pnt_type in test_types - copy_types:
                new_dyad = pnt_type | copy_type
                dyad_samps = new_dyad.get_samples(cdata.mtrees[lvls])

                if (dyad_samps > samp_dict[pnt_type]
                        and dyad_samps > samp_dict[copy_type]):
                    copy_dyads |= {new_dyad}
                    samp_dict[new_dyad] = dyad_samps

        lvl_types[lvls] = {
            mtype for mtype in test_types | copy_dyads
            if ((search_dict['samp_cutoff']
                 <= len(samp_dict[mtype]) <= max_samps)
                and samp_dict[mtype] != samp_dict[pnt_mtype])
            }

    test_mtypes = reduce(or_, lvl_types.values())
    rmv_mtypes = set()
    for rmv_mtype in sorted(test_mtypes):
        rmv_lvls = rmv_mtype.get_levels()

        # e.g. remove `Missense` in favour of `Missense->5th Exon` if
        # all of this gene's missense mutations are on the fifth exon
        for cmp_mtype in test_mtypes - {rmv_mtype} - rmv_mtypes:
            if (samp_dict[rmv_mtype] == samp_dict[cmp_mtype]
                    and (rmv_mtype.is_supertype(cmp_mtype)
                         or len(rmv_mtype.leaves()) > len(cmp_mtype.leaves())
                         or (len(rmv_lvls) > len(cmp_mtype.get_levels())))):
                rmv_mtypes |= {rmv_mtype}
                break

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

                expair_shal = ExMcomb(all_mtype - shal_mtype, mtype1, mtype2)
                exshal_samps = expair_shal.get_samples(cdata.mtrees[lvls])
                if exshal_samps != samp_dict[expair]:
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

