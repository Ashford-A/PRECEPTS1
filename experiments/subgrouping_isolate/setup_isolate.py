
from ..subgrouping_isolate.param_list import params
from .data_dirs import (firehose_dir, syn_root, metabric_dir, baml_dir,
                        gencode_dir, oncogene_list, subtype_file,
                        vep_cache_dir, expr_sources)

from dryadic.features.data.vep import process_variants
from ...features.cohorts.beatAML import (
    process_input_datasets as process_baml_datasets)
from ...features.cohorts.metabric import (
    process_input_datasets as process_metabric_datasets)
from ...features.cohorts.tcga import (
    process_input_datasets as process_tcga_datasets)

from ..utilities.mutations import (pnt_mtype, copy_mtype, shal_mtype,
                                   dup_mtype, gains_mtype, loss_mtype,
                                   dels_mtype, Mcomb, ExMcomb)
from dryadic.features.mutations import MuType
from ..subgrouping_isolate.utils import IsoMutationCohort

import os
import argparse
import synapseclient
import bz2
import dill as pickle

import pandas as pd
from itertools import combinations as combn


def get_input_datasets(cohort, expr_source,
                       use_genes=None, min_sources=2, mut_fields=None):
    data_dict = {data_k: None
                 for data_k in ('expr', 'vars', 'copy',
                                'annot', 'use_genes', 'assembly')}

    if use_genes is None:
        gene_df = pd.read_csv(oncogene_list,
                              sep='\t', skiprows=1, index_col=0)

        data_dict['use_genes'] = gene_df.index[
            (gene_df.loc[
                :, ['Vogelstein', 'SANGER CGC(05/30/2017)',
                    'FOUNDATION ONE', 'MSK-IMPACT']]
                == 'Yes').sum(axis=1) >= min_sources
            ].tolist()

    else:
        data_dict['use_genes'] = list(use_genes)

    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = syn_root
    syn.login()

    if cohort == 'beatAML':
        if expr_source != 'toil__gns':
            raise ValueError("Only gene-level Kallisto calls are available "
                             "for the beatAML cohort!")

        data_dict['assembly'] = 'GRCh37'

        data_dict.update({
            data_k: baml_data
            for data_k, baml_data in zip(
                ('expr', 'vars', 'copy', 'annot'),
                process_baml_datasets(baml_dir, gencode_dir, syn,
                                      annot_fields=['transcript'],
                                      mut_fields=mut_fields)
                )
            })

    elif cohort.split('_')[0] == 'METABRIC':
        if expr_source != 'microarray':
            raise ValueError("Only Illumina microarray mRNA calls are "
                             "available for the METABRIC cohort!")

        data_dict['assembly'] = 'GRCh37'

        if '_' in cohort:
            use_types = cohort.split('_')[1]
        else:
            use_types = None

        data_dict.update({
            data_k: mtbc_data
            for data_k, mtbc_data in zip(
                ('expr', 'vars', 'copy', 'annot'),
                process_metabric_datasets(
                    metabric_dir, gencode_dir, use_types,
                    annot_fields=['transcript'], mut_fields=mut_fields
                    )
                )
            })

    else:
        data_dict['assembly'] = 'GRCh37'

        source_info = expr_source.split('__')
        source_base = source_info[0]
        collapse_txs = not (len(source_info) > 1 and source_info[1] == 'txs')

        data_dict.update({
            data_k: tcga_data
            for data_k, tcga_data in zip(
                ('expr', 'vars', 'copy', 'annot'),
                process_tcga_datasets(
                    cohort, expr_source=source_base,
                    var_source='mc3', copy_source='Firehose',
                    expr_dir=expr_sources[source_base], annot_dir=gencode_dir,
                    type_file=subtype_file, collapse_txs=collapse_txs,
                    annot_fields=['transcript'], syn=syn,
                    mut_fields=mut_fields
                    )
                )
            })

    return data_dict


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

    args = parser.parse_args()
    out_path = os.path.join(args.out_dir, 'setup')

    data_dict = get_input_datasets(
        args.cohort, args.expr_source,
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

    # get the combination of mutation attributes and search constraints
    # that will govern the enumeration of subgroupings
    lvl_list = ('Gene', 'Scale', 'Copy') + tuple(args.mut_levels.split('__'))
    search_dict = params[args.search_params]

    # figure out which mutation attribute fields to request from VEP, starting
    # with those necessary to uniquely identify any mutation
    var_fields = ['Gene', 'Canonical', 'Location', 'VarAllele']
    for lvl in lvl_list[3:]:
        if '-domain' in lvl and 'Domains' not in var_fields:
            var_fields += ['Domains']
        else:
            var_fields += [lvl]

    # run the VEP command line wrapper to obtain a standardized
    # set of point mutation calls
    variants = process_variants(
        var_df, out_fields=var_fields, cache_dir=vep_cache_dir,
        temp_dir=out_path, assembly=data_dict['assembly'],
        distance=0, consequence_choose='pick', forks=4, update_cache=False
        )

    # remove mutation calls not assigned to a canonical transcript by VEP as
    # well as those not associated with genes linked to cancer processes
    variants = variants.loc[(variants.CANONICAL == 'YES')
                            & variants.Gene.isin(data_dict['use_genes'])]
    copies = data_dict['copy'].loc[
        data_dict['copy'].Gene.isin(data_dict['use_genes'])]

    assert not variants.duplicated().any(), (
        "Variant data contains {} duplicate entries!".format(
            variants.duplicated().sum())
        )

    cdata = IsoMutationCohort(data_dict['expr'], variants, [lvl_list], copies,
                              data_dict['annot'], leaf_annot=None)
    with bz2.BZ2File(os.path.join(out_path, "cohort-data.p.gz"), 'w') as f:
        pickle.dump(cdata, f, protocol=-1)

    max_samps = len(cdata.get_samples()) - search_dict['samp_cutoff']
    test_muts = set()

    # for each gene with enough point mutations, find all of the combinations
    # of its mutation subtypes that satisfy the search criteria
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

            if 'Copy' in dict(mtree):
                copy_types = {dup_mtype, loss_mtype}

                if 'ShalGain' in dict(mtree['Copy']):
                    copy_types |= {gains_mtype}
                if 'ShalDel' in dict(mtree['Copy']):
                    copy_types |= {dels_mtype}

            else:
                copy_types = set()

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

            test_types = pnt_types | copy_dyads
            if args.mut_levels == 'Consequence__Exon':
                test_types |= copy_types

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

    with open(os.path.join(out_path, "muts-list.p"), 'wb') as f:
        pickle.dump(sorted(test_muts), f, protocol=-1)
    with open(os.path.join(out_path, "muts-count.txt"), 'w') as fl:
        fl.write(str(len(test_muts)))


if __name__ == '__main__':
    main()

