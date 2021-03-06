
from ..utilities.mutations import (
    MuType, Mcomb, ExMcomb, shal_mtype, copy_mtype, gains_mtype, dels_mtype)
from ..utilities.metrics import calculate_mean_siml, calculate_ks_siml
from ..utilities.colour_maps import variant_clrs, mcomb_clrs
from ..utilities.labels import get_fancy_label

siml_fxs = {'mean': calculate_mean_siml, 'ks': calculate_ks_siml}
cna_mtypes = {'Gain': gains_mtype, 'Loss': dels_mtype}


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


def get_mcomb_lbl(mcomb):
    return '\n& '.join([
        get_fancy_label(tuple(mtype.subtype_iter())[0][1])
        for mtype in mcomb.mtypes
        ])

