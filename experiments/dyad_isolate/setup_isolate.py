
from ..dyad_isolate.param_lists import search_params, mut_lvls
from ..subgrouping_isolate.data_dirs import vep_cache_dir

from dryadic.features.data.vep import process_variants
from ..utilities.mutations import (pnt_mtype, shal_mtype,
                                   dup_mtype, loss_mtype, Mcomb, ExMcomb)
from dryadic.features.mutations import MuType

from ..subgrouping_isolate.setup_isolate import get_input_datasets
from ..subgrouping_isolate.utils import IsoMutationCohort

import os
import argparse
import bz2
import dill as pickle

import numpy as np
import pandas as pd

from itertools import combinations as combn
from itertools import product


def get_all_mtype(mtype, gene, use_mtrees, lvls_dict=None, base_lvls=None):
    sub_type = mtype.subtype_list()[0][1]

    if sub_type in lvls_dict:
        use_lvls = lvls_dict[mtype.subtype_list()[0][1]]
    elif base_lvls is not None:
        use_lvls = tuple(base_lvls)

    else:
        use_lvls = sorted(use_mtrees.keys())[0]

    return MuType({('Gene', gene): use_mtrees[use_lvls][gene].allkey()})


def main():
    parser = argparse.ArgumentParser(
        "Set up the paired-gene subtype expression effect isolation "
        "experiment by enumerating the subtypes to be tested."
        )

    parser.add_argument('expr_source', type=str,
                        help="a source of expression data")
    parser.add_argument('cohort', type=str, help="which TCGA cohort to use")
    parser.add_argument('search_params', type=str, choices=set(search_params))
    parser.add_argument('mut_lvls', type=str, choices=set(mut_lvls))
    parser.add_argument('out_dir', type=str)

    args = parser.parse_args()
    out_path = os.path.join(args.out_dir, 'setup')

    expr_data, mut_data, annot_dict, use_genes, use_asmb = get_input_datasets(
        args.cohort, args.expr_source, min_sources=3,
        mut_fields=['Sample', 'Gene', 'Chr', 'Start', 'End',
                    'Strand', 'RefAllele', 'TumorAllele']
        )

    var_data = mut_data.loc[mut_data.Scale == 'Point']
    copy_data = mut_data.loc[mut_data.Scale == 'Copy',
                             ['Sample', 'Gene', 'Scale', 'Copy']]

    var_df = pd.DataFrame({'Chr': var_data.Chr.astype('int'),
                           'Start': var_data.Start.astype('int'),
                           'End': var_data.End.astype('int'),
                           'RefAllele': var_data.RefAllele,
                           'VarAllele': var_data.TumorAllele,
                           'Strand': var_data.Strand,
                           'Sample': var_data.Sample})

    lvl_lists = [('Gene', 'Scale', 'Copy') + lvl_list
                 for lvl_list in mut_lvls[args.mut_lvls]]
    search_dict = search_params[args.search_params]

    var_fields = {'Gene', 'Canonical', 'Location', 'VarAllele'}
    for lvl_list in lvl_lists:
        for lvl in lvl_list[3:]:
            if '-domain' in lvl and 'Domains' not in var_fields:
                var_fields |= {'Domains'}
            else:
                var_fields |= {lvl}

    variants = process_variants(var_df, out_fields=var_fields,
                                cache_dir=vep_cache_dir,
                                temp_dir=out_path, assembly=use_asmb,
                                distance=0, consequence_choose='pick',
                                forks=4, update_cache=False)

    variants = variants.loc[variants.CANONICAL == 'YES']
    variants['Scale'] = 'Point'
    variants['Copy'] = np.nan

    use_muts = pd.concat([variants, copy_data], sort=True)
    use_muts = use_muts.loc[use_muts.Gene.isin(use_genes)]

    assert not use_muts.duplicated().any(), (
        "Variant data contains {} duplicate entries!".format(
            use_muts.duplicated().sum())
        )

    cdata = IsoMutationCohort(expr_data, use_muts, lvl_lists,
                              annot_dict, leaf_annot=None)
    with bz2.BZ2File(os.path.join(out_path, "cohort-data.p.gz"), 'w') as f:
        pickle.dump(cdata, f, protocol=-1)

    assert sorted(cdata.mtrees.keys()) == sorted(lvl_lists), (
        "Level combination mutation trees incorrectly instantiated!")

    mut_genes = set(tuple(sorted(dict(mtree)))
                    for mtree in cdata.mtrees.values())
    assert len(mut_genes) == 1, (
        "Level combination mutation trees contain mismatching sets of genes!")

    mut_genes = tuple(mut_genes)[0]
    total_samps = len(cdata.get_samples())
    max_samps = total_samps - search_dict['samp_cutoff']

    test_mtypes = dict()
    test_muts = set()
    lvls_dict = dict()

    for gene, mtree in cdata.mtrees[lvl_lists[0]]:
        root_types = {
            root_type for root_type in {pnt_mtype, dup_mtype, loss_mtype,
                                        pnt_mtype | dup_mtype,
                                        pnt_mtype | loss_mtype}
            if (len(root_type.get_samples(mtree))
                >= search_dict['samp_cutoff'])
            }

        samp_dict = {mtype: mtype.get_samples(mtree)
                     for mtype in root_types | {pnt_mtype}}
        pnt_types = set()

        if pnt_mtype in root_types:
            for lvls, lvl_tree in cdata.mtrees.items():
                lvl_types = lvl_tree[gene].combtypes(
                    mtype=pnt_mtype,
                    comb_sizes=tuple(
                        range(1, search_dict['branch_combs'] + 1)),
                    min_type_size=search_dict['samp_cutoff'],
                    min_branch_size=search_dict['min_branch']
                    )

                samp_dict.update({mtype: mtype.get_samples(lvl_tree[gene])
                                  for mtype in lvl_types})
                lvls_dict.update({mtype: lvls for mtype in lvl_types})
                pnt_types |= lvl_types

        gene_types = {mtype for mtype in pnt_types | root_types
                      if samp_dict[mtype] != samp_dict[pnt_mtype]}
        rmv_mtypes = set()

        for rmv_mtype in sorted(gene_types):
            rmv_lvls = rmv_mtype.get_levels()
            for cmp_mtype in sorted(gene_types - {rmv_mtype} - rmv_mtypes):
                cmp_lvls = cmp_mtype.get_levels()

                if (samp_dict[rmv_mtype] == samp_dict[cmp_mtype]
                        and (rmv_mtype.is_supertype(cmp_mtype)
                             or (any('domain' in lvl for lvl in rmv_lvls)
                                 and all('domain' not in lvl
                                         for lvl in cmp_lvls))
                             or len(rmv_lvls) < len(cmp_lvls)
                             or rmv_mtype > cmp_mtype)):
                    rmv_mtypes |= {rmv_mtype}
                    break

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
                          <= np.sum(cdata.train_pheno(mcomb)) <= max_samps)}

    with open(os.path.join(out_path, "muts-list.p"), 'wb') as f:
        pickle.dump(sorted(test_muts), f, protocol=-1)
    with open(os.path.join(out_path, "muts-count.txt"), 'w') as fl:
        fl.write(str(len(test_muts)))


if __name__ == '__main__':
    main()

