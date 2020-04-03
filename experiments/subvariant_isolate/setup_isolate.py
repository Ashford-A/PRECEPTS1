
import os
import sys
base_dir = os.path.dirname(__file__)
sys.path.extend([os.path.join(base_dir, '..', '..', '..')])

from HetMan.experiments.subvariant_isolate import *
from HetMan.experiments.subvariant_isolate.param_list import params

from dryadic.features.data.vep import process_variants
from HetMan.experiments.subvariant_test import (
    pnt_mtype, gain_mtype, loss_mtype)
from HetMan.experiments.subvariant_isolate import cna_mtypes

from HetMan.features.cohorts.beatAML import (
    process_input_datasets as process_baml_datasets)
from HetMan.features.cohorts.metabric import (
    process_input_datasets as process_metabric_datasets)
from HetMan.features.cohorts.tcga import (
    process_input_datasets as process_tcga_datasets)

from HetMan.experiments.subvariant_tour.utils import RandomType
from HetMan.experiments.subvariant_isolate.utils import Mcomb, ExMcomb
from dryadic.features.mutations import MuType, MuTree
from dryadic.features.cohorts.mut import BaseMutationCohort

import argparse
import synapseclient
import bz2
import dill as pickle

import numpy as np
import pandas as pd

import random
from itertools import combinations as combn
from itertools import product


def choose_source(cohort):
    # choose the source of expression data to use for this tumour cohort
    coh_base = cohort.split('_')[0]

    if coh_base == 'beatAML':
        use_src = 'toil__gns'
    elif coh_base in ['METABRIC', 'CCLE']:
        use_src = 'microarray'

    # default to using Broad Firehose expression calls for TCGA cohorts
    else:
        use_src = 'Firehose'

    return use_src


def get_input_datasets(cohort, mut_genes, mut_fields=None):
    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = syn_root
    syn.login()

    if cohort == 'beatAML':
        use_asmb = 'GRCh37'

        expr_data, mut_data, annot_dict = process_baml_datasets(
            baml_dir, annot_dir, syn,
            annot_fields=['transcript'], mut_fields=mut_fields
            )

    elif cohort.split('_')[0] == 'METABRIC':
        use_asmb = 'GRCh37'

        if '_' in cohort:
            use_types = cohort.split('_')[1]
        else:
            use_types = None

        expr_data, mut_data, annot_dict = process_metabric_datasets(
            metabric_dir, annot_dir, use_types,
            annot_fields=['transcript'], mut_fields=mut_fields
            )

    else:
        use_asmb = 'GRCh37'

        expr_data, mut_data, annot_dict = process_tcga_datasets(
            cohort, expr_source='Firehose', var_source='mc3',
            copy_source='Firehose', expr_dir=expr_dir, annot_dir=annot_dir,
            type_file=type_file, annot_fields=['transcript'], syn=syn,
            mut_fields=mut_fields
            )

    return expr_data, mut_data, annot_dict, use_asmb


def compare_lvls(lvls1, lvls2):
    for i in range(1, len(lvls1)):
        for j in range(1, len(lvls2) + 1):
            if lvls1[i:] == lvls2[:j]:
                return False

    for j in range(1, len(lvls2)):
        for i in range(1, len(lvls1) + 1):
            if lvls2[j:] == lvls1[:i]:
                return False

    return True


class IsoMutationCohort(BaseMutationCohort):

    def __add__(self, other):
        if not isinstance(other, IsoMutationCohort):
            return NotImplemented


def merge_cohorts(iso_cdatas):
    expr_hash = {iso_cdata.data_hash()[0] for iso_cdata in iso_cdatas}
    if len(expr_hash) > 1:
        raise ValueError("Cohorts have mismatching expression datasets!")

    annt_hash = {(tuple(iso_cdata.gene_annot), iso_cdata.leaf_annot)
                 for iso_cdata in iso_cdatas}
    if len(annt_hash) > 1:
        raise ValueError("Cohorts have mismatching annotation settings!")

    new_cdata = tuple(iso_cdatas)[0]
    for iso_cdata in iso_cdatas:
        for mut_lvls, mtree in iso_cdata.mtrees.items():
            if mut_lvls in new_cdata.mtrees:

                if hash(mtree) != hash(new_cdata.mtrees[mut_lvls]):
                    raise ValueError("Cohorts have mismatching mutation "
                                     "trees at levels `{}`!".format(mut_lvls))

            else:
                new_cdata.mtrees[mut_lvls] = mtree

        new_cdata.muts = new_cdata.muts.merge(
            iso_cdata.muts).drop_duplicates()

    for mut_lvls, mtree in new_cdata.mtrees.items():
        if hash(mtree) != hash(MuTree(new_cdata.muts, mut_lvls,
                                      leaf_annot=new_cdata.leaf_annot)):
            raise ValueError("Cohorts have internally inconsistent mutation "
                             "datasets at levels `{}`!".format(mut_lvls))

    return new_cdata


def main():
    parser = argparse.ArgumentParser(
        "Set up the gene subtype expression effect isolation experiment by "
        "enumerating the subtypes to be tested."
        )

    parser.add_argument('gene', type=str, help="a mutated gene")
    parser.add_argument('cohort', type=str, help="which TCGA cohort to use")
    parser.add_argument('search_params', type=str,)
    parser.add_argument('mut_levels', type=str, help="a mutated gene")
    parser.add_argument('out_dir', type=str)

    args = parser.parse_args()
    out_path = os.path.join(args.out_dir, 'setup')

    lvls_list = args.mut_levels.split('__')
    use_lvls = ['Scale', 'Copy'] + lvls_list
    use_params = params[args.search_params]

    expr_data, mut_data, annot_dict, use_asmb = get_input_datasets(
        args.cohort, [args.gene],
        mut_fields=['Sample', 'Gene', 'Chr', 'Start', 'End',
                    'Strand', 'RefAllele', 'TumorAllele']
        )

    cur_chr = annot_dict[args.gene]['Chr'].split('chr')[1]
    var_data = mut_data.loc[(mut_data.Scale == 'Point')
                            & (mut_data.Chr == cur_chr)]

    var_df = pd.DataFrame({'Chr': var_data.Chr.astype('int'),
                           'Start': var_data.Start.astype('int'),
                           'End': var_data.End.astype('int'),
                           'RefAllele': var_data.RefAllele,
                           'VarAllele': var_data.TumorAllele,
                           'Strand': var_data.Strand,
                           'Sample': var_data.Sample})

    var_fields = ['Gene']
    for lvl in lvls_list:
        if '-domain' in lvl and 'Domains' not in var_fields:
            var_fields += ['Domains']
        elif lvl not in ['Consequence', 'Position']:
            var_fields += [lvl]

    variants = process_variants(var_df, out_fields=var_fields,
                                cache_dir=vep_cache_dir, update_cache=True,
                                temp_dir=out_path, assembly=use_asmb,
                                forks=4, distance=0)

    variants = variants.loc[(variants.Gene == args.gene)
                            & (variants.CANONICAL == 'YES')]
    variants['Scale'] = 'Point'
    variants['Copy'] = np.nan

    gene_muts = mut_data.loc[(mut_data.Scale == 'Copy')
                             & (mut_data.Gene == args.gene)]
    gene_muts = pd.concat([variants, gene_muts], sort=True)

    cdata = IsoMutationCohort(expr_data, gene_muts, [use_lvls], [args.gene],
                              annot_dict, leaf_annot=None)
    with bz2.BZ2File(os.path.join(out_path, "cohort-data.p.gz"), 'w') as f:
        pickle.dump(cdata, f, protocol=-1)

    max_samps = len(cdata.get_samples()) - use_params['samp_cutoff']
    use_mtree = cdata.mtrees[tuple(use_lvls)]
    brnch_mtypes = use_mtree.branchtypes(mtype=pnt_mtype,
                                         min_size=use_params['samp_cutoff'])

    comb_types = use_mtree.combtypes(
        mtype=pnt_mtype,
        comb_sizes=tuple(range(1, use_params['branch_combs'] + 1)),
        min_type_size=use_params['samp_cutoff'],
        min_branch_size=use_params['min_branch']
        )

    samp_dict = {mtype: mtype.get_samples(use_mtree) for mtype in comb_types}
    pnt_samps = use_mtree['Point'].get_samples()
    pnt_types = {mtype for mtype in comb_types
                  if samp_dict[mtype] != pnt_samps}

    print("found {} potential point mutations...".format(len(pnt_types)))

    rmv_mtypes = set()
    for rmv_mtype in sorted(pnt_types):
        rmv_lvls = rmv_mtype.get_levels()

        for cmp_mtype in pnt_types - {rmv_mtype} - rmv_mtypes:
            if (samp_dict[rmv_mtype] == samp_dict[cmp_mtype]
                    and (rmv_mtype.is_supertype(cmp_mtype)
                         or len(rmv_lvls) < len(cmp_mtype.get_levels()))):
                rmv_mtypes |= {rmv_mtype}
                break

    pnt_types -= rmv_mtypes
    print("after filtering: {} point mutations...".format(len(pnt_types)))

    if 'Copy' in dict(use_mtree):
        copy_types = {gain_mtype, loss_mtype}

        if 'ShalGain' in dict(use_mtree['Copy']):
            copy_types |= {dict(cna_mtypes)['Gain']}
        if 'ShalDel' in dict(use_mtree['Copy']):
            copy_types |= {dict(cna_mtypes)['Loss']}

    else:
        copy_types = set()

    copy_dyads = set()
    for copy_type in copy_types:
        samp_dict[copy_type] = copy_type.get_samples(use_mtree)

        if len(samp_dict[copy_type]) >= 5:
            for pnt_type in pnt_types:
                new_dyad = pnt_type | copy_type
                dyad_samps = new_dyad.get_samples(use_mtree)

                if (dyad_samps > samp_dict[pnt_type]
                        and dyad_samps > samp_dict[copy_type]):
                    copy_dyads |= {new_dyad}
                    samp_dict[new_dyad] = dyad_samps

    print("found {} potential copy dyads...".format(len(copy_dyads)))
    test_types = {
        mtype for mtype in pnt_types | copy_types | copy_dyads
        if use_params['samp_cutoff'] <= len(samp_dict[mtype]) <= max_samps
        }

    print("after filtering: {} total mutation types...".format(len(test_types)))

    """
    # TODO: double-check that these are uniquely generated
    use_mtypes |= {
        RandomType(size_dist=int(np.sum(cdata.train_pheno(mtype))),
                   base_mtype=pnt_mtype, seed=(i + 3) * j + 93307)
        for i, mtype in enumerate(test_mtypes) for j in range(98, 102)
        }

    max_size = max(np.sum(cdata.train_pheno(mtype))
                   for mtype in conf_df.index) + base_size
    max_size = int(max_size / 2)

    use_mtypes |= {RandomType(size_dist=(use_ctf, max_size), seed=seed + 39)
                   for seed in range((max_size - use_ctf) * 2)}
    use_mtypes |= {
        RandomType(size_dist=(use_ctf, max_size),
                   base_mtype=MuType({('Gene', args.gene): pnt_mtype}),
                   seed=seed + 79103)
        for seed in range((max_size - use_ctf) * 2)
        }
    """

    all_mtype = MuType(use_mtree.allkey())
    ex_mtypes = [MuType({}), dict(cna_mtypes)['Shal']]
    mtype_lvls = {mtype: mtype.get_levels() - {'Scale'}
                  for mtype in pnt_types | copy_types}

    use_pairs = {(mtype2, mtype1) if 'Copy' in lvls1 else (mtype1, mtype2)
                 for (mtype1, lvls1), (mtype2, lvls2)
                 in combn(mtype_lvls.items(), 2)
                 if (('Copy' not in lvls1 or 'Copy' not in lvls2)
                     and (mtype1 & mtype2).is_empty()
                     and not samp_dict[mtype1] >= samp_dict[mtype2]
                     and not samp_dict[mtype2] >= samp_dict[mtype1])}
    print("found {} potential comb pairs...".format(len(use_pairs)))

    use_mcombs = {Mcomb(*pair) for pair in use_pairs}
    use_mcombs |= {ExMcomb(all_mtype - ex_mtype, *pair)
                   for pair in use_pairs for ex_mtype in ex_mtypes}

    use_mcombs |= {ExMcomb(all_mtype - ex_mtype, mtype)
                   for mtype in test_types for ex_mtype in ex_mtypes
                   if not isinstance(mtype, RandomType)}

    test_combs = {mcomb for mcomb in use_mcombs
                  if (use_params['samp_cutoff']
                      <= len(mcomb.get_samples(use_mtree)) <= max_samps)}
    print("after filtering: {} total combs...".format(len(test_combs)))

    with open(os.path.join(out_path, "muts-list.p"), 'wb') as f:
        pickle.dump(sorted(test_types | test_combs), f, protocol=-1)
        with open(os.path.join(out_path, "muts-count.txt"), 'w') as fl:
            fl.write(str(len(test_types) + len(test_combs)))


if __name__ == '__main__':
    main()

