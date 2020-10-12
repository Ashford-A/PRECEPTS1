
params = {
    'default': {
        'samp_cutoff': 20, 'branch_combs': 3, 'min_branch': 5},

    'deep': {
        'samp_cutoff': 20, 'branch_combs': 4, 'min_branch': 4},

    'semideep': {
        'samp_cutoff': 20, 'branch_combs': 2, 'min_branch': 5},

    'shallow': {
        'samp_cutoff': 20, 'branch_combs': 2, 'min_branch': 10},

    }

mut_lvls = {
    'default': (('Consequence', 'Exon'),
                ('Exon', 'Position', 'HGVSp'),
                ('SMART-domains', 'Consequence'),
                ('Pfam-domain', 'Consequence')),

    'exons': (('Exon', 'Consequence'), ),
    }

