
import os
import sys
base_dir = os.path.dirname(__file__)
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.experiments.subvariant_transfer import *
from HetMan.experiments.utilities.load_input import parse_subtypes
from HetMan.features.cohorts.tcga import MutationCohort, MutationConcatCohort
from dryadic.features.mutations import MuType
from HetMan.experiments.subvariant_infer.setup_infer import Mcomb, ExMcomb

import argparse
import synapseclient
import pandas as pd
import dill as pickle

from functools import reduce
from operator import or_, and_
from itertools import combinations as combn
from itertools import product


def get_cohorts(expr_source, cohorts, mut_levels, cv_prop=1.0, cv_seed=None):
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

    source_info = expr_source.split('__')
    source_base = source_info[0]
    collapse_txs = not (len(source_info) > 1 and source_info[1] == 'txs')
    cohorts_base = {cohort: cohort.split('_')[0] for cohort in cohorts}

    cdata_dict = {
        cohort: MutationCohort(
            cohort=cohorts_base[cohort], mut_genes=use_genes.tolist(),
            mut_levels=['Gene'] + mut_levels, expr_source=source_base,
            var_source='mc3', copy_source='Firehose', annot_file=annot_file,
            type_file=type_file, expr_dir=expr_sources[expr_source],
            copy_dir=copy_dir, collapse_txs=collapse_txs,
            syn=syn, cv_prop=cv_prop, cv_seed=cv_seed,
            annot_fields=['transcript'], use_types=parse_subtypes(cohort)
            )
        for cohort in cohorts
        }

    cdata = MutationConcatCohort(
        cohorts=list(cohorts_base.values()), mut_genes=use_genes.tolist(),
        mut_levels=['Gene'] + mut_levels, expr_source=source_base,
        var_source='mc3', copy_source='Firehose', annot_file=annot_file,
        type_file=type_file, expr_dir=expr_sources[expr_source],
        copy_dir=copy_dir, collapse_txs=collapse_txs, syn=syn,
        cv_prop=cv_prop, cv_seed=cv_seed, annot_fields=['transcript'],
        use_types={cohorts_base[cohort]: parse_subtypes(cohort)
                   for cohort in cohorts}
        )

    return cdata, cdata_dict


def main():
    parser = argparse.ArgumentParser(
        "Set up the gene subtype expression effect isolation experiment by "
        "enumerating the subtypes to be tested."
        )

    # create positional command line arguments
    parser.add_argument('mut_levels', type=str,
                        help="the mutation property levels to consider")
    parser.add_argument('cohorts', type=str, nargs='+',
                        help="which TCGA cohort to use")

    # create optional command line arguments
    parser.add_argument('--samp_cutoff', type=int, default=20,
                        help='subtype sample frequency threshold')
    parser.add_argument('--setup_dir', type=str, default=base_dir)

    # parse command line arguments
    args = parser.parse_args()
    out_path = os.path.join(args.setup_dir, 'setup')
    use_lvls = args.mut_levels.split('__')
    cdata_concat, cdata_dict = get_cohorts('Firehose', args.cohorts, use_lvls)

    with open(os.path.join(out_path, "cohort-data.p"), 'wb') as f:
        pickle.dump(cdata_concat, f)
    with open(os.path.join(out_path, "cohort-dict.p"), 'wb') as f:
        pickle.dump(cdata_dict, f)

    mtype_list = {cohort: set() for cohort in args.cohorts}
    for cohort, cdata in cdata_dict.items():
        for gene, muts in cdata.train_mut:
            
            use_mtypes = {
                mtype for mtype in (muts.branchtypes(min_size=20)
                                    - {MuType({('Scale', 'Copy'): None})}
                                    - {MuType({('Scale', 'Copy'): {
                                        ('Copy', 'ShalGain'): None}})}
                                    - {MuType({('Scale', 'Copy'): {
                                        ('Copy', 'ShalDel'): None}})})
                if (20 <= len(mtype.get_samples(muts))
                    <= (len(cdata.samples) - 20))
                }

            if args.mut_levels != 'Location__Protein':
                use_mtypes -= {MuType({('Scale', 'Point'): None})}

            use_pairs = {(mtype1, mtype2)
                         for mtype1, mtype2 in combn(use_mtypes, 2)
                         if (mtype1 & mtype2).is_empty()}
            use_mcombs = {Mcomb(*pair) for pair in use_pairs}
            use_mcombs |= {ExMcomb(muts, *pair) for pair in use_pairs}

            if args.mut_levels != 'Location__Protein':
                use_mtypes = {
                    mtype for mtype in use_mtypes
                    if (mtype & MuType({('Scale', 'Copy'): None})).is_empty()
                    }

            use_mtypes -= {mtype1
                           for mtype1, mtype2 in product(use_mtypes, repeat=2)
                           if (mtype1 != mtype2
                               and mtype1.is_supertype(mtype2)
                               and (mtype1.get_samples(cdata.train_mut)
                                    == mtype2.get_samples(cdata.train_mut)))}

            use_mcombs |= {ExMcomb(muts, mtype) for mtype in use_mtypes}
            use_mtypes |= {mcomb for mcomb in use_mcombs
                           if (20
                               <= len(mcomb.get_samples(muts))
                               <= (len(cdata.samples) - 20))}

            mtype_list[cohort] |= {MuType({('Gene', gene): mtype})
                                   for mtype in use_mtypes}

    train_mtypes = [
        (cohort, mtype)
        for cohort, mtypes in mtype_list.items() for mtype in mtypes
        if (len(mtype.get_samples(cdata_dict[cohort].train_mut))
            >= args.samp_cutoff
            and any(other_mtype.subtype_list()[0][0]
                    == mtype.subtype_list()[0][0]
                    for other_mtype in mtype_list[cohort] - {mtype})
            and mtype in reduce(
                or_, [oth_mtypes
                      for oth_cohort, oth_mtypes in mtype_list.items()
                      if oth_cohort != cohort]
                ))
        ]

    pickle.dump(sorted(train_mtypes),
                open(os.path.join(out_path, "muts-list.p"), 'wb'))
    with open(os.path.join(out_path, "muts-count.txt"), 'w') as fl:
        fl.write(str(len(train_mtypes)))


if __name__ == '__main__':
    main()

