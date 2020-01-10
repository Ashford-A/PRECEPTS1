
import os
import sys
base_dir = os.path.dirname(__file__)
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.experiments.subvariant_threshold import *
from HetMan.experiments.subvariant_test.setup_test import (
    get_cohort_data, load_cohort)
from HetMan.experiments.subvariant_infer.setup_infer import choose_source
from HetMan.features.cohorts.tcga import list_cohorts

from HetMan.experiments.subvariant_threshold.utils import MutThresh
from HetMan.experiments.subvariant_test import (
    pnt_mtype, copy_mtype, gain_mtype, loss_mtype)
from HetMan.experiments.subvariant_tour.utils import RandomType
from dryadic.features.mutations import MuType

import argparse
from pathlib import Path
import bz2
import dill as pickle
import subprocess

import numpy as np
import pandas as pd

from functools import reduce
from operator import or_
import random


def main():
    parser = argparse.ArgumentParser(
        "Set up the gene subtype expression effect isolation experiment by "
        "enumerating the subtypes to be tested."
        )

    parser.add_argument('cohort', type=str, help="which TCGA cohort to use")
    parser.add_argument('classif', type=str, help="a mutation classifier")
    parser.add_argument('out_dir', type=str, default=base_dir)
    parser.add_argument('test_dir', type=str, default=base_dir)

    args = parser.parse_args()
    use_coh = args.cohort.split('_')[0]
    use_source = choose_source(use_coh)
 
    base_path = os.path.join(
        args.out_dir.split('subvariant_threshold')[0], 'subvariant_threshold')
    coh_dir = os.path.join(base_path, 'setup')
    coh_path = os.path.join(coh_dir, "cohort-data__{}.p".format(args.cohort))
    out_path = os.path.join(args.out_dir, 'setup')

    # find all the subvariant enumeration experiments that have run to
    # completion using the given combination of cohort and mutation classifier
    test_outs = Path(os.path.join(args.test_dir, 'subvariant_test')).glob(
        os.path.join("{}__{}__samps-*".format(use_source, args.cohort),
                     "trnsf-vals__*__{}.p.gz".format(args.classif))
        )

    # parse the enumeration experiment output files to find the minimum sample
    # occurence threshold used for each mutation annotation level tested
    out_datas = [Path(out_file).parts[-2:] for out_file in test_outs]
    out_df = pd.DataFrame([{'Samps': int(out_data[0].split('__samps-')[1]),
                            'Levels': '__'.join(out_data[1].split(
                                'trnsf-vals__')[1].split('__')[:-1])}
                           for out_data in out_datas])

    if 'Exon__Location__Protein' not in set(out_df.Levels):
        raise ValueError("Cannot infer subvariant behaviour until the "
                         "`subvariant_test` experiment is run "
                         "with mutation levels `Exon__Location__Protein` "
                         "which tests genes' base mutations!")

    # load bootstrapped AUCs for enumerated subgrouping mutations
    conf_dict = dict()
    for lvls, ctf in out_df.groupby('Levels')['Samps']:
        conf_fl = os.path.join(
            args.test_dir, 'subvariant_test',
            "{}__{}__samps-{}".format(use_source, args.cohort, ctf.values[0]),
            "out-conf__{}__{}.p.gz".format(lvls, args.classif)
            )

        with bz2.BZ2File(conf_fl, 'r') as f:
            conf_dict[lvls] = pickle.load(f)['mean']

    conf_vals = pd.concat(conf_dict.values())
    conf_vals = conf_vals[[not isinstance(mtype, RandomType)
                           for mtype in conf_vals.index]]

    test_genes = {'Point': set(), 'Gain': set(), 'Loss': set()}
    for gene, conf_vec in conf_vals.groupby(
            lambda mtype: mtype.get_labels()[0]):

        if len(conf_vec) > 1:
            auc_vec = conf_vec.apply(np.mean)
            base_mtype = MuType({('Gene', gene): pnt_mtype})
            base_indx = auc_vec.index.get_loc(base_mtype)

            sub_aucs = auc_vec[:base_indx].append(
                auc_vec[(base_indx + 1):])
            best_subtype = sub_aucs.idxmax()
            best_indx = auc_vec.index.get_loc(best_subtype)

            if auc_vec[best_indx] > 0.65:
                test_genes['Point'] |= {gene}

                for test_mtype in sub_aucs.index:
                    mtype_sub = test_mtype.subtype_list()[0][1]

                    if ((gene not in test_genes['Gain'])
                            and not (mtype_sub & gain_mtype).is_empty()):
                        test_indx = 'Gain'

                    elif ((gene not in test_genes['Loss'])
                            and not (mtype_sub & loss_mtype).is_empty()):
                        test_indx = 'Loss'

                    else:
                        test_indx = None

                    if test_indx is not None:
                        conf_sc = np.greater.outer(
                            conf_vec[test_mtype],
                            conf_vec[base_mtype]
                            ).mean()

                        if conf_sc > 0.75:
                            test_genes[test_indx] |= {gene}

    use_genes = list(reduce(or_, test_genes.values()))
    use_mtypes = set()
    use_ctf = int(out_df.Samps.min())
    mtree_k = ['Gene', 'Scale', 'Copy']
    use_lfs = ('ref_count', 'alt_count', 'PolyPhen', 'SIFT', 'depth')

    cdata = get_cohort_data(args.cohort, choose_source(args.cohort),
                            [mtree_k], use_genes, leaf_annot=use_lfs,
                            gene_annot=['transcript'])

    with open(coh_path, 'wb') as f:
        pickle.dump(cdata, f, protocol=-1)
    with open(os.path.join(out_path, "cohort-data.p"), 'wb') as f:
        pickle.dump(cdata, f, protocol=-1)

    for gene, mtree in cdata.mtrees[tuple(mtree_k)]:
        base_mtypes = {MuType({('Gene', gene): pnt_mtype})}

        if gene in test_genes['Gain']:
            base_mtypes |= {MuType({('Gene', gene): pnt_mtype | gain_mtype})}
        if gene in test_genes['Loss']:
            base_mtypes |= {MuType({('Gene', gene): pnt_mtype | loss_mtype})}

        for base_mtype in base_mtypes:
            base_size = len(base_mtype.get_samples(
                cdata.mtrees[tuple(mtree_k)]))

            gene_mtypes = {
                MutThresh('VAF', vaf_val, base_mtype)
                for vaf_val in set(
                    max(alt_cnt / (alt_cnt + ref_cnt)
                        for alt_cnt, ref_cnt in zip(vals['alt_count'],
                                                    vals['ref_count']))
                    for vals in pnt_mtype.get_leaf_annot(
                        mtree, ['ref_count', 'alt_count']).values()
                    )
                }

            test_mtype = MutThresh(
                'VAF', 0.25, MuType({('Gene', gene): pnt_mtype | gain_mtype}))
            n_samps = len(test_mtype.get_samples(cdata.mtrees[tuple(mtree_k)]))

            for lf_annt in ['PolyPhen', 'SIFT', 'depth']:
                gene_mtypes |= {
                    MutThresh(lf_annt, annt_val, base_mtype)
                    for annt_val in set(max(vals[lf_annt])
                                        for vals in pnt_mtype.get_leaf_annot(
                                            mtree, [lf_annt]).values())
                    if annt_val > 0
                    }

            use_mtypes |= {
                mtype for mtype in gene_mtypes
                if (use_ctf
                    <= len(mtype.get_samples(cdata.mtrees[tuple(mtree_k)]))
                    < min(base_size, len(cdata.get_samples()) - use_ctf + 1))
                }

    with open(os.path.join(out_path, "muts-list.p"), 'wb') as f:
        pickle.dump(sorted(use_mtypes), f, protocol=-1)
    with open(os.path.join(out_path, "muts-count.txt"), 'w') as fl:
        fl.write(str(len(use_mtypes)))

    coh_list = list_cohorts('Firehose', expr_dir=expr_dir, copy_dir=copy_dir)
    coh_list -= {args.cohort}
    use_feats = set(cdata.get_features())
    random.seed()

    for coh in random.sample(coh_list, k=len(coh_list)):
        coh_base = coh.split('_')[0]
        coh_tag = "cohort-data__{}.p".format(coh)
        coh_path = os.path.join(coh_dir, coh_tag)

        trnsf_cdata = load_cohort(
            coh, 'Firehose', coh_path, mut_lvls=[mtree_k],
            use_genes=use_genes, leaf_annot=use_lfs, gene_annot=['transcript']
            )

        use_feats &= set(trnsf_cdata.get_features())
        with open(coh_path, 'wb') as f:
            pickle.dump(trnsf_cdata, f, protocol=-1)

        copy_prc = subprocess.run(['cp', coh_path,
                                   os.path.join(out_path, coh_tag)],
                                  check=True)

    with open(os.path.join(out_path, "feat-list.p"), 'wb') as f:
        pickle.dump(use_feats, f, protocol=-1)


if __name__ == '__main__':
    main()

