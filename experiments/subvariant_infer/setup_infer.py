
import os
import sys
base_dir = os.path.dirname(__file__)
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.experiments.subvariant_infer import *
from HetMan.experiments.subvariant_infer.utils import (
    Mcomb, ExMcomb, RandomType)
from HetMan.experiments.subvariant_infer.merge_infer import merge_cohort_data

from HetMan.experiments.subvariant_tour.utils import calculate_aucs
from HetMan.experiments.utilities.load_input import load_firehose_cohort
from HetMan.features.cohorts.tcga import list_cohorts
from HetMan.features.cohorts.beatAML import BeatAmlCohort
from dryadic.features.mutations import MuType

import argparse
import synapseclient
from glob import glob
from pathlib import Path
import pandas as pd
import bz2
import dill as pickle

import random
from functools import reduce
from operator import and_
from itertools import combinations as combn
from itertools import product


def main():
    parser = argparse.ArgumentParser(
        "Set up the gene subtype expression effect isolation experiment by "
        "enumerating the subtypes to be tested."
        )

    parser.add_argument('cohort', type=str, help="which TCGA cohort to use")
    parser.add_argument('gene', type=str, help="which gene to consider")
    parser.add_argument('classif', type=str, help="which gene to consider")

    parser.add_argument('out_dir', type=str, default=base_dir)
    parser.add_argument('tour_dir', type=str, default=base_dir)

    args = parser.parse_args()
    if args.cohort == 'beatAML':
        use_source = 'toil__gns'
    else:
        use_source = 'Firehose'

    tour_outs = glob(os.path.join(args.tour_dir, 'subvariant_tour',
                                  "{}__{}__samps-*".format(
                                      use_source, args.cohort),
                                  "out-data__*__{}.p.gz".format(
                                      args.classif)))

    out_datas = [Path(out_file).parts[-2:] for out_file in tour_outs]
    out_use = pd.DataFrame([{'Samps': int(out_data[0].split('__samps-')[1]),
                             'Levels': '__'.join(out_data[1].split(
                                 'out-data__')[1].split('__')[:-1])}
                            for out_data in out_datas]).groupby(
                                ['Levels'])['Samps'].min()

    if 'Exon__Location__Protein' not in out_use.index:
        raise ValueError("Cannot infer subvariant behaviour until the "
                         "`subvariant_tour` experiment is run "
                         "with mutation levels `Exon__Location__Protein` "
                         "which tests genes' base mutations!")

    cdata_dict = {
        lvls: merge_cohort_data(os.path.join(
            args.tour_dir, 'subvariant_tour', "{}__{}__samps-{}".format(
                use_source, args.cohort, ctf)
            ), lvls, use_seed=8713)
        for lvls, ctf in out_use.iteritems()
        }

    use_genes = set(tuple(cdata.mut_genes) for cdata in cdata_dict.values())
    assert len(use_genes) == 1, ("Differing sets of mutated genes in the "
                                 "same cohort across different mutation "
                                 "annotation levels!")
    use_genes = tuple(use_genes)[0]

    if args.gene not in use_genes:
        raise ValueError("No mutations were tested for gene "
                         "{} in `{}`!".format(args.gene, args.tour_dir))

    infer_dict = {
        lvls: pickle.load(bz2.BZ2File(os.path.join(
            args.tour_dir, 'subvariant_tour', "{}__{}__samps-{}".format(
                use_source, args.cohort, ctf),
            "out-data__{}__{}.p.gz".format(lvls, args.classif)
            ), 'r'))['Infer']['Chrm']
        for lvls, ctf in out_use.iteritems()
        }

    out_dict = {lvls: calculate_aucs({'Chrm': infer_df}, cdata_dict[lvls])
                for lvls, infer_df in infer_dict.items()}
    pheno_dict = {mtype: phn for _, phn_dict in out_dict.values()
                  for mtype, phn in phn_dict.items()}

    auc_vals = pd.concat([auc_df['Chrm'] for auc_df, _ in out_dict.values()])
    auc_vals = auc_vals.loc[[mtype.subtype_list()[0][0] == args.gene
                             for mtype in auc_vals.index]]

    if auc_vals.shape[0] == 0:
        raise ValueError("No mutations were tested for gene "
                         "{} in `{}`!".format(args.gene, args.tour_dir))

    base_mtype = MuType({('Gene', args.gene): {('Scale', 'Point'): None}})
    good_indx = auc_vals >= auc_vals[base_mtype]

    random.seed(5001)
    use_mtypes = {lvls: {
        MuType({('Gene', args.gene): {
            ('Scale', 'Point'): mtype.subtype_list()[0][1]}})
        for mtype in sorted(set(auc_vals.index) - {base_mtype})
        if ((mtype in infer_dict[lvls].index and (good_indx[mtype]
                                                  or random.random() <= 0.1))
            or mtype.subtype_list()[0][1] in cdata_dict[lvls].mtree[
                args.gene]['Point'].branchtypes(min_size=ctf))
        } for lvls, ctf in sorted(out_use.iteritems())}

    max_size = 0
    use_mtypes['Random'] = set()

    for lvls, cdata in cdata_dict.items():
        mtype_sizes = [len(mtype.get_samples(cdata.mtree))
                       for mtype in use_mtypes[lvls]]
        max_size = max(max_size, max(mtype_sizes))

        use_mtypes['Random'] |= {
            RandomType(size_dist=mtype_size,
                       base_mtype=MuType({('Gene', args.gene): {(
                           'Scale', 'Point'): None}}),
                       seed=seed)
            for mtype_size, seed in product(mtype_sizes, range(89, 99))
            }

    max_size = int(
        (max_size + len(cdata.mtree[args.gene]['Point'].get_samples())) / 2)

    use_mtypes['Random'] |= {
        RandomType(size_dist=[int(out_use.min()), max_size], seed=seed)
        for seed in range((max_size - out_use.min()) * 2)
        }
    use_mtypes['Random'] |= {
        RandomType(size_dist=[int(out_use.min()), max_size],
                   base_mtype=MuType({('Gene', args.gene): {(
                       'Scale', 'Point'): None}}),
                   seed=seed)
        for seed in range((max_size - out_use.min()) * 2)
        }

    use_mtypes['Exon__Location__Protein'] |= {base_mtype}
    copy_mtree = cdata_dict[out_use.index[0]].mtree[args.gene]['Copy']
    samp_count = len(cdata_dict[out_use.index[0]].get_samples())

    copy_mtypes = copy_mtree.branchtypes(min_size=out_use.min())
    copy_mtypes |= {MuType({('Copy', ('DeepGain', 'ShalGain')): None})}
    copy_mtypes |= {MuType({('Copy', ('DeepDel', 'ShalDel')): None})}

    copy_mtypes -= {
        mtype1 for mtype1, mtype2 in product(copy_mtypes, repeat=2)
        if mtype1 != mtype2 and mtype1.is_supertype(mtype2)
        and (mtype1.get_samples(copy_mtree) == mtype2.get_samples(copy_mtree))
        }

    use_mtypes['Copy'] = {
        MuType({('Gene', args.gene): {('Scale', 'Copy'): mtype}})
        for mtype in copy_mtypes if (out_use.min()
                                     <= len(mtype.get_samples(copy_mtree))
                                     <= (samp_count - out_use.min()))
        }

    for lvls in out_use.index:
        use_mtree = cdata_dict[lvls].mtree

        use_pairs = {(mtype1, mtype2)
                     for mtype1, mtype2 in combn(use_mtypes[lvls]
                                                 | use_mtypes['Copy'], 2)
                     if (mtype1 & mtype2).is_empty()}

        use_mcombs = {Mcomb(*pair) for pair in use_pairs}
        use_mcombs |= {ExMcomb(use_mtree, *pair) for pair in use_pairs}
        use_mcombs |= {ExMcomb(use_mtree, mtype)
                       for mtype in use_mtypes[lvls]}

        if lvls == 'Exon__Location__Protein':
            use_mcombs |= {ExMcomb(use_mtree, mtype)
                           for mtype in use_mtypes['Copy']}

        use_mtypes[lvls] |= {mcomb for mcomb in use_mcombs
                             if (out_use.min()
                                 <= len(mcomb.get_samples(use_mtree))
                                 <= (samp_count - out_use.min()))}

    mtype_list = [(lvls, mtype) for lvls, mtypes in use_mtypes.items()
                  for mtype in mtypes]

    base_path = os.path.join(args.out_dir.split('subvariant_infer')[0],
                             'subvariant_infer')
    coh_path = os.path.join(base_path, 'setup')
    out_path = os.path.join(args.out_dir, 'setup')

    coh_list = list_cohorts('Firehose', expr_dir=expr_dir, copy_dir=copy_dir)
    coh_list |= {args.cohort, 'beatAML'}

    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = syn_root
    syn.login()

    use_feats = None
    for coh in random.sample(coh_list, k=len(coh_list)):
        coh_tag = "{}__cohort-dict.p".format(coh)

        if coh == args.cohort:
            copy_tag = "cohort-dict.p"
        else:
            copy_tag = "{}__cohort-dict.p".format(coh)

        if os.path.exists(os.path.join(coh_path, coh_tag)):
            with open(os.path.join(coh_path, coh_tag), 'rb') as f:
                cdata_dict = pickle.load(f)

            for lvls in set(out_use.index) - set(cdata_dict.keys()):
                if coh == 'beatAML':
                    if 'Domain' not in lvls:
                        cdata_dict[lvls] = BeatAmlCohort(
                            ['Gene'] + lvls.split('__'), use_genes,
                            expr_source='toil__gns',
                            expr_file=beatAML_files['expr'],
                            samp_file=beatAML_files['samps'], syn=syn,
                            annot_file=annot_file, domain_dir=domain_dir,
                            cv_seed=709, test_prop=0
                            )

                else:
                    cdata_dict[lvls] = load_firehose_cohort(
                        coh, use_genes, ['Gene'] + lvls.split('__'),
                        cv_seed=709, test_prop=0
                        )

        else:
            if coh == 'beatAML':
                cdata_dict = {
                    lvls: BeatAmlCohort(['Gene'] + lvls.split('__'),
                                        use_genes, expr_source='toil__gns',
                                        expr_file=beatAML_files['expr'],
                                        samp_file=beatAML_files['samps'],
                                        syn=syn, annot_file=annot_file,
                                        domain_dir=domain_dir,
                                        cv_seed=709, test_prop=0)
                    for lvls in out_use.index if 'Domain' not in lvls
                    }

            else:
                cdata_dict = {
                    lvls: load_firehose_cohort(coh, use_genes,
                                               ['Gene'] + lvls.split('__'),
                                               cv_seed=709, test_prop=0)
                    for lvls in out_use.index
                    }

        if use_feats is None:
            use_feats = reduce(and_, [set(cdata.get_features())
                                      for cdata in cdata_dict.values()])

        else:
            use_feats &= reduce(and_, [set(cdata.get_features())
                                       for cdata in cdata_dict.values()])

        with open(os.path.join(coh_path, coh_tag), 'wb') as f:
            pickle.dump(cdata_dict, f)
        with open(os.path.join(out_path, copy_tag), 'wb') as f:
            pickle.dump(cdata_dict, f)

    with open(os.path.join(out_path, "feat-list.p"), 'wb') as f:
        pickle.dump(use_feats, f)
    with open(os.path.join(out_path, "muts-list.p"), 'wb') as f:
        pickle.dump(sorted(mtype_list), f)
    with open(os.path.join(out_path, "muts-count.txt"), 'w') as fl:
        fl.write(str(len(mtype_list)))


if __name__ == '__main__':
    main()

