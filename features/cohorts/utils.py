
cohort_subtypes = {
    'nonbasal': ['LumA', 'LumB', 'Her2', 'Normal'],
    'luminal': ['LumA', 'LumB'],
    }


def parse_subtypes(cohort):
    use_subtypes = None
    coh_info = cohort.split('_')

    if len(coh_info) > 1:
        if coh_info[1] in cohort_subtypes:
            use_subtypes = cohort_subtypes[coh_info[1]]
        else:
            use_subtypes = [coh_info[1]]

    return use_subtypes

