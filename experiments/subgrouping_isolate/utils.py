
from ..utilities.mutations import (MuType, Mcomb, ExMcomb, deep_mtype,
                                   shal_mtype, copy_mtype, gains_mtype,
                                   dels_mtype, dup_mtype, loss_mtype)
from ..utilities.metrics import calculate_mean_siml, calculate_ks_siml
from ..utilities.colour_maps import variant_clrs, mcomb_clrs
from ..utilities.labels import get_fancy_label
from ..subgrouping_isolate import base_dir, train_cohorts

import numpy as np
import pandas as pd
import bz2
import dill as pickle
from pathlib import Path
from zipfile import ZipFile

siml_fxs = {'mean': calculate_mean_siml, 'ks': calculate_ks_siml}
cna_mtypes = {'Iso': {'All': shal_mtype | deep_mtype,
                      'Gain': [gains_mtype, dup_mtype],
                      'Loss': [dels_mtype, loss_mtype]},
              'IsoShal': {'All': deep_mtype,
                          'Gain': [dup_mtype], 'Loss': [loss_mtype]}}


# TODO: is this still useful without pre-computed similarities?
def search_siml_pair(siml_dicts, mut, other_mut):
    simls = dict()

    for mut_lvls, siml_dfs in siml_dicts.items():
        for siml_df in siml_dfs:
            if mut in siml_df.columns and other_mut in siml_df.index:
                simls[mut_lvls] = siml_df.loc[other_mut, mut]
                break

    return simls


def remove_pheno_dups(muts, pheno_dict):
    mut_phns = set()
    mut_list = set()

    for mut in muts:
        mut_phn = tuple(pheno_dict[mut].tolist())

        if mut_phn not in mut_phns:
            mut_phns |= {mut_phn}
            mut_list |= {mut}

    return mut_list


def get_mut_ex(mut):
    if isinstance(mut, ExMcomb):
        if (mut.all_mtype & shal_mtype).is_empty():
            mut_ex = 'IsoShal'
        else:
            mut_ex = 'Iso'

    elif isinstance(mut, (MuType, Mcomb)):
        mut_ex = 'All'

    else:
        raise TypeError(
            "Unrecognized type of mutation <{}>!".format(type(mut)))

    return mut_ex


def choose_subtype_colour(mut):
    if (copy_mtype & mut).is_empty():
        mut_clr = variant_clrs['Point']

    elif gains_mtype.is_supertype(mut):
        mut_clr = variant_clrs['Gain']
    elif dels_mtype.is_supertype(mut):
        mut_clr = variant_clrs['Loss']

    elif not (gains_mtype & mut).is_empty():
        mut_clr = mcomb_clrs['Point+Gain']
    elif not (dels_mtype & mut).is_empty():
        mut_clr = mcomb_clrs['Point+Loss']

    return mut_clr


def get_mcomb_lbl(mcomb, **lbl_kwargs):
    return '\n& '.join([
        get_fancy_label(tuple(mtype.subtype_iter())[0][1], **lbl_kwargs)
        for mtype in sorted(mcomb.mtypes)
        ])


def load_cohorts_data(out_list, ex_lbl, data_cache=None, load_objs=None):
    if data_cache and Path(data_cache).exists():
        objs = [None, None, None, None]

        if load_objs is None:
            load_objs = ['preds', 'phns', 'aucs', 'cdatas']

        for i, obj_lbl in enumerate(['preds', 'phns', 'aucs', 'cdatas']):
            if obj_lbl in load_objs:
                with ZipFile(data_cache, 'r') as f:
                    objs[i] = pickle.loads(f.read(obj_lbl))

        pred_dfs, phn_dicts, auc_lists, cdata_dict = objs

    else:
        pred_dfs, phn_dicts, auc_lists, cdata_dict = load_cohorts_output(
            out_list, ex_lbl)

        if data_cache:
            with ZipFile(data_cache, 'w') as f:
                f.writestr('preds', pickle.dumps(pred_dfs, protocol=-1))

            for lbl, obj in zip(['phns', 'aucs', 'cdatas'],
                                [phn_dicts, auc_lists, cdata_dict]):
                with ZipFile(data_cache, 'a') as f:
                    f.writestr(lbl, pickle.dumps(obj, protocol=-1))

    return pred_dfs, phn_dicts, auc_lists, cdata_dict


def load_cohorts_output(out_list, ex_lbl):
    use_iter = out_list.groupby(['Source', 'Cohort', 'Levels'])['File']

    out_dirs = {(src, coh): Path(base_dir, '__'.join([src, coh]))
                for src, coh, _ in use_iter.groups}
    out_tags = {fl: '__'.join(fl.parts[-1].split('__')[1:])
                for fl in out_list.File}
    pred_tag = "out-pred_{}".format(ex_lbl)

    phn_dicts = {(src, coh): dict() for src, coh, _ in use_iter.groups}
    cdata_dict = {(src, coh): None for src, coh, _ in use_iter.groups}

    auc_lists = {(src, coh): pd.Series(dtype='float')
                 for src, coh, _ in use_iter.groups}
    pred_dfs = {(src, coh): pd.DataFrame() for src, coh, _ in use_iter.groups}

    for (src, coh, lvls), out_files in use_iter:
        out_aucs = list()
        out_preds = list()

        for out_file in out_files:
            with bz2.BZ2File(Path(out_dirs[src, coh],
                                  '__'.join(["out-pheno",
                                             out_tags[out_file]])),
                             'r') as f:
                phn_dicts[src, coh].update(pickle.load(f))

            with bz2.BZ2File(Path(out_dirs[src, coh],
                                  '__'.join(["out-aucs",
                                             out_tags[out_file]])),
                             'r') as f:
                out_aucs += [pickle.load(f)[ex_lbl]['mean']]

            with bz2.BZ2File(Path(out_dirs[src, coh],
                                  '__'.join([pred_tag, out_tags[out_file]])),
                             'r') as f:
                pred_vals = pickle.load(f)

            out_preds += [pred_vals.applymap(np.mean)]

            with bz2.BZ2File(Path(out_dirs[src, coh],
                                  '__'.join(["cohort-data",
                                             out_tags[out_file]])),
                             'r') as f:
                new_cdata = pickle.load(f)

            if cdata_dict[src, coh] is None:
                cdata_dict[src, coh] = new_cdata
            else:
                cdata_dict[src, coh].merge(new_cdata)

        mtypes_comp = np.greater_equal.outer(
            *([[set(auc_vals.index) for auc_vals in out_aucs]] * 2))
        super_comp = np.apply_along_axis(all, 1, mtypes_comp)

        # if there is not a subgrouping set that contains all the others,
        # concatenate the output of all sets...
        if not super_comp.any():
            auc_lists[src, coh] = auc_lists[src, coh].append(
                pd.concat(out_aucs, sort=False))
            pred_dfs[src, coh] = pd.concat(
                [pred_dfs[src, coh], *out_preds], sort=False)

        # ...otherwise, use the "superset"
        else:
            super_indx = super_comp.argmax()

            auc_lists[src, coh] = auc_lists[src, coh].append(
                out_aucs[super_indx])
            pred_dfs[src, coh] = pd.concat(
                [pred_dfs[src, coh], out_preds[super_indx]], sort=False)

    # filter out duplicate subgroupings due to overlapping search criteria
    for src, coh, _ in use_iter.groups:
        auc_lists[src, coh].sort_index(inplace=True)
        pred_dfs[src, coh].sort_index(inplace=True)
        assert (auc_lists[src, coh].index == pred_dfs[src, coh].index).all()

        auc_lists[src, coh] = auc_lists[src, coh].loc[
            ~auc_lists[src, coh].index.duplicated()]
        pred_dfs[src, coh] = pred_dfs[src, coh].loc[
            ~pred_dfs[src, coh].index.duplicated()]

    return pred_dfs, phn_dicts, auc_lists, cdata_dict
