
def remove_pair_dups(mut_pairs, pheno_dict):
    pair_infos = set()
    pair_list = set()

    for mut1, mut2 in mut_pairs:
        pair_info = tuple(sorted([tuple(pheno_dict[mut1]),
                                  tuple(pheno_dict[mut2])]))
        pair_info += tuple(sorted(set(mut1.label_iter())
                                  | set(mut2.label_iter())))

        if pair_info not in pair_infos:
            pair_infos |= {pair_info}
            pair_list |= {(mut1, mut2)}

    return pair_list

