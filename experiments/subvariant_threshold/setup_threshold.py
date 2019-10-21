
import os
import sys
base_dir = os.path.dirname(__file__)
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.experiments.subvariant_threshold import *
from HetMan.experiments.subvariant_tour.setup_tour import get_cohort_data
from HetMan.experiments.subvariant_tour.utils import RandomType
from HetMan.experiments.subvariant_infer.setup_infer import choose_source
from HetMan.features.cohorts.tcga import list_cohorts

from HetMan.experiments.subvariant_threshold.utils import MutThresh
from HetMan.experiments.subvariant_tour import pnt_mtype
from dryadic.features.mutations import MuType

import argparse
from pathlib import Path
import bz2
import dill as pickle
import subprocess

import pandas as pd
import random


def main():
    parser = argparse.ArgumentParser(
        "Set up the gene subtype expression effect isolation experiment by "
        "enumerating the subtypes to be tested."
        )

    parser.add_argument('cohort', type=str, help="which TCGA cohort to use")
    parser.add_argument('classif', type=str, help="which gene to consider")
    parser.add_argument('out_dir', type=str, default=base_dir)
    parser.add_argument('tour_dir', type=str, default=base_dir)

    args = parser.parse_args()
    use_coh = args.cohort.split('_')[0]
    use_source = choose_source(use_coh)
 
    base_path = os.path.join(
        args.out_dir.split('subvariant_threshold')[0], 'subvariant_threshold')
    coh_path = os.path.join(base_path, 'setup')
    out_path = os.path.join(args.out_dir, 'setup')

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

    conf_vals = pd.concat(conf_dict.values())
    conf_vals = conf_vals[[not isinstance(mtype, RandomType)
                           for mtype in conf_vals.index]].iloc[:, 0]

    if conf_vals.shape[0] == 0:
        raise ValueError("No mutations were tested for gene "
                         "{} in `{}`!".format(args.gene, args.tour_dir))

    use_genes = list({
        mtype.get_labels()[0] for mtype in conf_vals.groupby(
            lambda mtype: mtype.get_labels()[0]).filter(
                lambda vals: vals.shape[0] > 1).index
        })

    use_lfs = ('ref_count', 'alt_count', 'PolyPhen', 'SIFT', 'depth')
    cdata = get_cohort_data(args.cohort, choose_source(args.cohort),
                            [['Gene', 'Scale']], use_genes,
                            leaf_annot=use_lfs, gene_annot=['transcript'])

    use_mtypes = set()
    use_ctf = int(out_df.Samps.min())

    for gene, mtree in cdata.mtrees['Gene', 'Scale']:
        base_mtype = MuType({('Gene', gene): pnt_mtype})
        base_size = len(base_mtype.get_samples(cdata.mtrees['Gene', 'Scale']))

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
                <= len(mtype.get_samples(cdata.mtrees['Gene', 'Scale']))
                < min(base_size, len(cdata.get_samples()) - use_ctf + 1))
            }

    with open(os.path.join(out_path, "muts-list.p"), 'wb') as f:
        pickle.dump(sorted(use_mtypes), f, protocol=-1)
    with open(os.path.join(out_path, "muts-count.txt"), 'w') as fl:
        fl.write(str(len(use_mtypes)))

    coh_list = list_cohorts('Firehose', expr_dir=expr_dir, copy_dir=copy_dir)
    coh_list |= {args.cohort}
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
                                              mut_lvls=[['Gene', 'Scale']],
                                              leaf_annot=use_lfs,
                                              gene_annot=['transcript'])

        else:
            trnsf_cdata = get_cohort_data(coh, choose_source(coh),
                                          mut_lvls=[['Gene', 'Scale']],
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

