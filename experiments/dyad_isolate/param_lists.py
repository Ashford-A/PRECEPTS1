
search_params = {
    'default': {'samp_cutoff': 25,
                'branch_combs': 1,
                'min_branch': 25},

    'deep': {'samp_cutoff': 25,
             'branch_combs': 2,
             'min_branch': 10},

    'shallow': {'samp_cutoff': 40,
                'branch_combs': 1,
                'min_branch': 40},
    }

mut_lvls = {
    'default': [['Exon', 'Position', 'HGVSp'],
                ['Consequence', 'Exon'],
                ['Pfam-domain', 'Consequence'],
                ['SMART-domains', 'Consequence']],

    'form_base': [['Form_base', 'Exon']],
    }

