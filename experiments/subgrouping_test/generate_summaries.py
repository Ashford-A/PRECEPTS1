
from ..utilities.mutations import copy_mtype, pnt_mtype, RandomType
from dryadic.features.mutations import MuType

from ..subgrouping_test import base_dir
from ..utilities.misc import get_label, get_subtype
from ..utilities.labels import get_fancy_label
from ..utilities.metrics import calc_delong

import os
import argparse
from pathlib import Path
import bz2
import dill as pickle
from operator import itemgetter

import numpy as np
import pandas as pd


use_lvls = ['Consequence__Exon', 'Exon__Position__HGVSp',
            'Pfam-domain__Consequence', 'SMART-domains__Consequence']
summ_dir = os.path.join(base_dir, 'summaries')


def get_table_gene(mtype):
    if isinstance(mtype, RandomType):
        if mtype.base_mtype is None:
            gn = 'n/a'
        else:
            gn = get_label(mtype.base_mtype)
    else:
        gn = get_label(mtype)

    return gn


def get_table_label(mtype):
    if isinstance(mtype, RandomType):
        lbl = repr(mtype)
    else:
        lbl = get_fancy_label(get_subtype(mtype))

    return lbl


def main():
    parser = argparse.ArgumentParser(
        'generate_summaries',
        description="Creates tables condensing output of experiments."
        )

    parser.add_argument('classif', help="a mutation classifier")
    args = parser.parse_args()

    out_datas = [
        out_file.parts[-2:] for out_file in Path(base_dir).glob(os.path.join(
            "*__*__samps-*", "out-trnsf__*__{}.p.gz".format(args.classif)))
        ]

    out_list = pd.DataFrame([
        {'Source': '__'.join(out_data[0].split('__')[:-2]),
         'Cohort': out_data[0].split('__')[-2],
         'Samps': int(out_data[0].split("__samps-")[1]),
         'Levels': '__'.join(out_data[1].split(
             "out-trnsf__")[1].split('__')[:-1])}
        for out_data in out_datas
        ]).groupby('Cohort').filter(
        lambda outs: all(lvl in set(outs.Levels) for lvl in use_lvls))

    if out_list.shape[0] == 0:
        raise ValueError("No experiment output found for these parameters!")

    out_use = out_list.groupby(['Source', 'Cohort', 'Levels'])['Samps'].min()
    coh_iter = tuple(out_use.groupby(['Source', 'Cohort']))[::-1]
    os.makedirs(os.path.join(summ_dir, args.classif), exist_ok=True)

    for (src, coh), outs in coh_iter:
        print(coh)
        phn_dict = dict()

        pred_df = pd.DataFrame()
        auc_df = pd.DataFrame()
        coef_df = pd.DataFrame()
        trnsf_df = pd.DataFrame()

        for (_, _, lvls), ctf in outs.iteritems():
            out_tag = "{}__{}__samps-{}".format(src, coh, ctf)

            with bz2.BZ2File(
                    os.path.join(base_dir, out_tag,
                                 "out-pheno__{}__{}.p.gz".format(
                                     lvls, args.classif)),
                    'r') as f:
                phn_dict.update(pickle.load(f))

            with bz2.BZ2File(
                    os.path.join(base_dir, out_tag,
                                 "out-pred__{}__{}.p.gz".format(
                                     lvls, args.classif)),
                    'r') as f:
                pred_vals = pickle.load(f)
                pred_df = pred_df.append(pred_vals.applymap(np.mean))

            with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                          "out-aucs__{}__{}.p.gz".format(
                                              lvls, args.classif)),
                             'r') as f:
                auc_df = auc_df.append(pickle.load(f))

            with bz2.BZ2File(
                    os.path.join(base_dir, out_tag,
                                 "out-trnsf__{}__{}.p.gz".format(
                                     lvls, args.classif)),
                    'r') as f:
                trnsf_data = pickle.load(f)

            if trnsf_data:
                trnsf_mat = pd.DataFrame({
                    'AUC': pd.DataFrame({
                        coh: trnsf_out['AUC']['mean']
                        for coh, trnsf_out in trnsf_data.items()
                        if trnsf_out['AUC'].shape[0] > 0
                        }).unstack().dropna().round(4)
                    })

                trnsf_mat = trnsf_mat.assign(
                    Size=[sum(trnsf_data[coh]['Pheno'][mtype])
                          for coh, mtype in trnsf_mat.index]
                    )
                trnsf_df = trnsf_df.append(trnsf_mat)

            with bz2.BZ2File(
                    os.path.join(base_dir, out_tag,
                                 "out-coef__{}__{}.p.gz".format(
                                     lvls, args.classif)),
                    'r') as f:
                coef_mat = pickle.load(f)

            coef_df = coef_df.append(
                coef_mat.groupby(level=0, axis=1).mean().applymap(
                    lambda coef: format(coef, '.3g'))
                )

        assert (sorted(phn_dict) == sorted(auc_df.index)
                == sorted(coef_df.index))

        mtype_dict = {
            'Base': sorted(mtype for mtype in phn_dict
                           if not isinstance(mtype, RandomType)
                           and (get_subtype(mtype) & copy_mtype).is_empty()),

            'Copy': sorted(mtype for mtype in phn_dict
                           if not isinstance(mtype, RandomType)
                           and not (get_subtype(mtype)
                                    & copy_mtype).is_empty()),

            'RandCoh': sorted(mtype for mtype in phn_dict
                              if (isinstance(mtype, RandomType)
                                  and mtype.base_mtype is None)),

            'RandGene': sorted(mtype for mtype in phn_dict
                               if (isinstance(mtype, RandomType)
                                   and mtype.base_mtype is not None)),

            }

        print({mtype_lbl: len(mtypes)
               for mtype_lbl, mtypes in mtype_dict.items()})
        mtype_dict = {mtype_lbl: mtypes
                      for mtype_lbl, mtypes in mtype_dict.items() if mtypes}

        for mtype_lbl, mtypes in mtype_dict.items():
            mtype_tbl = pd.DataFrame({
                'Subgrouping': [str(mtype) for mtype in mtypes],
                'Gene': [get_table_gene(mtype) for mtype in mtypes],
                'Label': [get_table_label(mtype) for mtype in mtypes],
                'Sample Count': [sum(phn_dict[mtype]) for mtype in mtypes]
                })

            mtype_tbl.to_csv(
                os.path.join(summ_dir, args.classif,
                             "{}__{}__mtype-tbl_{}.csv".format(
                                 src, coh, mtype_lbl)),
                index=False
                )

            auc_mat = pd.DataFrame({
                'Subgrouping': [str(mtype) for mtype in mtypes],
                'AUC_mean': auc_df.loc[mtypes, 'mean'].round(4)
                })

            for i in range(10):
                auc_mat['AUC_cv{}'.format(i + 1)] = auc_df.loc[
                    mtypes, 'CV'].apply(itemgetter(i))

            if mtype_lbl != 'RandCoh':
                base_infrs = {
                    gene: pred_df.loc[
                        MuType({('Gene', gene): pnt_mtype})].values
                    for gene in {get_table_gene(mtype)
                                 for mtype in auc_mat.index}
                    }

                auc_mat = auc_mat.assign(pDivg=[
                    format(calc_delong(pred_df.loc[mtype].values,
                                       base_infrs[get_table_gene(mtype)],
                                       phn_dict[mtype],
                                       auc_df.loc[mtype, 'mean']), '.3e')

                    if (isinstance(mtype, RandomType)
                        or get_subtype(mtype) != pnt_mtype) else 1.
                    for mtype in mtypes
                    ])

            pred_df.loc[mtypes].round(5).to_csv(
                os.path.join(summ_dir, args.classif,
                             "{}__{}__pred-vals_{}.csv".format(
                                 src, coh, mtype_lbl)),
                )

            auc_mat.to_csv(
                os.path.join(summ_dir, args.classif,
                             "{}__{}__auc-mat_{}.csv".format(
                                 src, coh, mtype_lbl)),
                index=False
                )

            if mtype_lbl in {'Base', 'Copy'} and trnsf_df.shape[0] > 0:
                trnsf_mat = trnsf_df.loc[[
                    mtype in mtypes for _, mtype in trnsf_df.index]]

                trnsf_mat = trnsf_mat.assign(
                    Index=[mtypes.index(mtype)
                           for _, mtype in trnsf_mat.index]
                    )
                trnsf_mat.index = trnsf_mat.index.set_names([
                    'Cohort', 'Subgrouping'])

                trnsf_mat.sort_index().reset_index().to_csv(
                    os.path.join(summ_dir, args.classif,
                                 "{}__{}__trnsf-aucs_{}.csv".format(
                                     src, coh, mtype_lbl)),
                    index=False
                    )

            coef_df.loc[mtypes].to_csv(
                os.path.join(summ_dir, args.classif,
                             "{}__{}__coef-means_{}.csv".format(
                                 src, coh, mtype_lbl))
                )


if __name__ == '__main__':
    main()

