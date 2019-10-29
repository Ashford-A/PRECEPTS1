
import os
import sys
base_dir = os.path.dirname(__file__)
sys.path.extend([os.path.join(base_dir, '..', '..', '..')])

from HetMan.experiments.subvariant_infer import *
from HetMan.experiments.subvariant_infer import gain_mtype, loss_mtype
from HetMan.features.cohorts.tcga import list_cohorts
from HetMan.experiments.subvariant_tour.setup_tour import get_cohort_data

from HetMan.experiments.subvariant_tour.utils import RandomType
from HetMan.experiments.subvariant_tour import pnt_mtype
from HetMan.experiments.subvariant_infer.utils import Mcomb, ExMcomb
from dryadic.features.mutations import MuType

import argparse
from pathlib import Path
import bz2
import dill as pickle
import subprocess

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

    # parse command line arguments, get mutation type representing all
    # point mutations of the given gene
    args = parser.parse_args()
    use_coh = args.cohort.split('_')[0]
    use_source = choose_source(use_coh)
    base_mtype = MuType({('Gene', args.gene): pnt_mtype})

    # find all the subvariant enumeration experiments that have run to
    # completion using the given combination of cohort and mutation classifier
    tour_outs = Path(os.path.join(args.tour_dir, 'subvariant_tour')).glob(
        os.path.join("{}__{}__samps-*".format(use_source, args.cohort),
                     "out-conf__*__{}.p.gz".format(args.classif))
        )

    # parse the enumeration experiment output files to find the minimum sample
    # occurence threshold used for each mutation annotation level tested
    out_datas = [Path(out_file).parts[-2:] for out_file in tour_outs]
    out_df = pd.DataFrame([{'Samps': int(out_data[0].split('__samps-')[1]),
                            'Levels': '__'.join(out_data[1].split(
                                'out-conf__')[1].split('__')[:-1])}
                           for out_data in out_datas])

    if 'Exon__Location__Protein' not in set(out_df.Levels):
        raise ValueError("Cannot infer subvariant behaviour until the "
                         "`subvariant_tour` experiment is run "
                         "with mutation levels `Exon__Location__Protein` "
                         "which tests genes' base mutations!")

    # load bootstrapped AUCs for enumerated subgrouping mutations
    conf_dict = dict()
    for lvls, ctf in out_df.groupby('Levels')['Samps']:
        conf_fl = os.path.join(
            args.tour_dir, 'subvariant_tour',
            "{}__{}__samps-{}".format(use_source, args.cohort, ctf.values[0]),
            "out-conf__{}__{}.p.gz".format(lvls, args.classif)
            )

        with bz2.BZ2File(conf_fl, 'r') as f:
            conf_dict[lvls] = pickle.load(f)['Chrm']

    # consolidate bootstrapped AUCs and filter out AUCs for mutations
    # representing random sets of samples rather than actual mutations
    conf_vals = pd.concat(conf_dict.values())
    conf_vals = conf_vals[[(not isinstance(mtype, RandomType)
                            and mtype.get_labels()[0] == args.gene)
                           for mtype in conf_vals.index]]

    if conf_vals.shape[0] == 0:
        raise ValueError("No mutations were tested for gene "
                         "{} in `{}`!".format(args.gene, args.tour_dir))

    conf_df = pd.DataFrame.from_items(zip(conf_vals.index,
                                          conf_vals.values[:, 0])).transpose()

    use_lvls = [
        list(lvls) for lvls in (
            {mtype.get_sorted_levels() for mtype in conf_df.index}
            | {('Gene', 'Exon', 'Location', 'Protein')}
            )
        ]

    use_lfs = 'ref_count', 'alt_count', 'PolyPhen'
    cdata = get_cohort_data(
        args.cohort, choose_source(args.cohort), use_lvls, [args.gene],
        leaf_annot=use_lfs, gene_annot=['transcript']
        )

    mtype_lvlk = {mtype: cdata.choose_mtree(mtype) for mtype in conf_df.index}
    main_lvls = ('Gene', 'Scale', 'Copy', 'Exon', 'Location', 'Protein')

    base_confs = conf_df.loc[base_mtype].values
    conf_scores = {mtype: np.greater.outer(confs.values, base_confs).mean()
                   for mtype, confs in conf_df.iterrows()}
    base_size = len(cdata.mtrees[main_lvls][args.gene]['Point'].get_samples())

    use_ctf = int(out_df.Samps.min())
    brnch_mtypes = {
        lvls: (mtree[args.gene]['Point'].branchtypes(min_size=use_ctf)
               if lvls not in [('Gene', 'Scale', 'Copy'), ('Gene', 'Scale')]
               else set())
        for lvls, mtree in cdata.mtrees.items()
        }

    # select an enumerated subgrouping for further testing if it represents
    # all point mutations, a mutation of a single annotation branch, or
    # according to a probability proportional to the number of bootstrapped
    # AUCs greater than the bootstrapped AUCs of the parent point mutation
    random.seed(5001)
    use_mtypes = {
        MuType({('Gene', args.gene): {
            ('Scale', 'Point'): mtype.subtype_list()[0][1]}})
        if mtype != base_mtype else base_mtype
        for mtype, conf_sc in sorted(conf_scores.items())
        if (mtype == base_mtype
            or (np.sum(cdata.train_pheno(mtype)) < base_size
                and (conf_sc >= random.random()
                     or (mtype.subtype_list()[0][1]
                         in brnch_mtypes[mtype_lvlk[mtype]]))))
        }

    copy_dyads = set()
    if len(gain_mtype.get_samples(cdata.mtrees[main_lvls][args.gene])) >= 5:
        copy_dyads |= {mtype | MuType({('Gene', args.gene): gain_mtype})
                       for mtype in use_mtypes}
    if len(loss_mtype.get_samples(cdata.mtrees[main_lvls][args.gene])) >= 5:
        copy_dyads |= {mtype | MuType({('Gene', args.gene): loss_mtype})
                       for mtype in use_mtypes}

    # TODO: double-check that these are uniquely generated
    use_mtypes |= {
        RandomType(size_dist=int(np.sum(cdata.train_pheno(mtype))),
                   base_mtype=MuType({('Gene', args.gene): pnt_mtype}),
                   seed=(i + 3) * j + 93307)
        for i, mtype in enumerate(use_mtypes) for j in range(98, 102)
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

    copy_mtree = cdata.mtrees[main_lvls][args.gene]['Copy']
    copy_mtypes = copy_mtree.branchtypes(min_size=use_ctf)
    copy_mtypes |= {MuType({('Copy', ('DeepGain', 'ShalGain')): None})}
    copy_mtypes |= {MuType({('Copy', ('DeepDel', 'ShalDel')): None})}

    copy_mtypes -= {
        mtype1 for mtype1, mtype2 in product(copy_mtypes, repeat=2)
        if mtype1 != mtype2 and mtype1.is_supertype(mtype2)
        and (mtype1.get_samples(copy_mtree) == mtype2.get_samples(copy_mtree))
        }

    use_mtypes |= {
        MuType({('Gene', args.gene): {('Scale', 'Copy'): mtype}})
        for mtype in copy_mtypes if (use_ctf
                                     <= len(mtype.get_samples(copy_mtree))
                                     <= (len(cdata.get_samples()) - use_ctf))
        }

    all_mtypes = {
        lvls: MuType({('Gene', args.gene): MuType(mtree[args.gene].allkey())})
        for lvls, mtree in cdata.mtrees.items()
        }
    ex_mtypes = [MuType({}), MuType({('Gene', args.gene): {
        ('Scale', 'Copy'): {('Copy', ('ShalGain', 'ShalDel')): None}}})]

    mtype_lvls = {mtype: mtype.get_sorted_levels()[2:]
                  for mtype in use_mtypes
                  if not isinstance(mtype, RandomType)}

    mcomb_lvls = {mtype: (('Gene', 'Scale') if not lvls
                          else main_lvls if lvls == ('Copy', )
                          else ('Gene', 'Scale', 'Copy') + lvls)
                  for mtype, lvls in mtype_lvls.items()}

    use_pairs = {(mtype2, mtype1) if 'Copy' in lvls1 else (mtype1, mtype2)
                 for (mtype1, lvls1), (mtype2, lvls2)
                 in combn(mtype_lvls.items(), 2)
                 if ((('Copy' in lvls1) ^ ('Copy' in lvls2))
                     and (set(lvls1) & set(lvls2)) in [set(lvls1), set(lvls2)]
                     and compare_lvls(lvls1, lvls2)
                     and (mtype1 & mtype2).is_empty())}

    use_mcombs = {Mcomb(*pair) for pair in use_pairs}
    use_mcombs |= {ExMcomb(all_mtypes[mcomb_lvls[pair[0]]] - ex_mtype, *pair)
                   for pair in use_pairs for ex_mtype in ex_mtypes}

    use_mcombs |= {ExMcomb(all_mtypes[mcomb_lvls[mtype]] - ex_mtype, mtype)
                   for mtype in use_mtypes for ex_mtype in ex_mtypes
                   if not isinstance(mtype, RandomType)}

    use_mtypes |= {mcomb for mcomb in use_mcombs
                   if (use_ctf <= np.sum(cdata.train_pheno(mcomb))
                       <= (len(cdata.get_samples()) - use_ctf))}
    use_mtypes |= {mtype for mtype in copy_dyads
                   if (use_ctf <= np.sum(cdata.train_pheno(mtype))
                       <= (len(cdata.get_samples()) - use_ctf))}

    base_path = os.path.join(args.out_dir.split('subvariant_infer')[0],
                             'subvariant_infer')
    coh_path = os.path.join(base_path, args.gene, 'setup')
    out_path = os.path.join(args.out_dir, 'setup')

    with open(os.path.join(out_path, "muts-list.p"), 'wb') as f:
        pickle.dump(sorted(use_mtypes), f, protocol=-1)
    with open(os.path.join(out_path, "muts-count.txt"), 'w') as fl:
        fl.write(str(len(use_mtypes)))

    coh_list = list_cohorts('Firehose', expr_dir=expr_dir, copy_dir=copy_dir)
    coh_list |= {args.cohort, 'beatAML', 'METABRIC', 'CCLE'}
    use_feats = set(cdata.get_features())

    random.seed()
    for coh in random.sample(coh_list, k=len(coh_list)):
        coh_tag = "{}__cohort-data.p".format(coh)

        if coh == args.cohort:
            copy_tag = "cohort-data.p"
        else:
            copy_tag = "{}__cohort-data.p".format(coh)

        if os.path.exists(os.path.join(coh_path, coh_tag)):
            try:
                with open(os.path.join(coh_path, coh_tag), 'rb') as f:
                    trnsf_cdata = pickle.load(f)

            except EOFError:
                trnsf_cdata = get_cohort_data(coh, choose_source(coh),
                                              mut_lvls=use_lvls,
                                              use_genes=[args.gene],
                                              leaf_annot=use_lfs,
                                              gene_annot=['transcript'])

        else:
            trnsf_cdata = get_cohort_data(coh, choose_source(coh),
                                          mut_lvls=use_lvls,
                                          use_genes=[args.gene],
                                          leaf_annot=use_lfs,
                                          gene_annot=['transcript'])

        use_feats &= set(trnsf_cdata.get_features())
        with open(os.path.join(coh_path, coh_tag), 'wb') as f:
            pickle.dump(trnsf_cdata, f, protocol=-1)

        copy_prc = subprocess.run(
            ['cp', os.path.join(coh_path, coh_tag),
             os.path.join(out_path, copy_tag)], check=True
            )

    with open(os.path.join(out_path, "feat-list.p"), 'wb') as f:
        pickle.dump(use_feats, f, protocol=-1)


if __name__ == '__main__':
    main()

