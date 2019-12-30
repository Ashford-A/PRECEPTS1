
import os
import sys
base_dir = os.path.dirname(__file__)
sys.path.extend([os.path.join(base_dir, '..', '..', '..')])

from HetMan.experiments.subvariant_tour import *
from HetMan.experiments.subvariant_tour.param_list import params, mut_lvls
from HetMan.experiments.subvariant_test import pnt_mtype
from HetMan.experiments.subvariant_test.setup_test import load_cohort
from dryadic.features.mutations import MuType

import argparse
import dill as pickle
from itertools import product


def main():
    parser = argparse.ArgumentParser(
        "Set up the gene subtype expression effect isolation experiment by "
        "enumerating the subtypes to be tested."
        )

    parser.add_argument('expr_source', type=str,
                        help="which TCGA expression data source to use")
    parser.add_argument('cohort', type=str, help="which TCGA cohort to use")
    parser.add_argument('search_params', type=str,)
    parser.add_argument('mut_lvls', type=str,)
    parser.add_argument('--out_dir', type=str, default=base_dir)

    # parse command line arguments
    args = parser.parse_args()
    out_path = os.path.join(args.out_dir, 'setup')
    use_params = params[args.search_params]
    lvls_list = mut_lvls[args.mut_lvls]

    coh_path = os.path.join(args.out_dir.split('subvariant_tour')[0],
                            'subvariant_tour',
                            '__'.join([args.expr_source, args.cohort]),
                            "cohort-data.p")

    cdata = load_cohort(args.cohort, args.expr_source, coh_path)
    n_samps = len(cdata.get_samples())

    for use_lvls in lvls_list:
        lbls_key = ('Gene', 'Scale') + tuple(use_lvls)
        if lbls_key not in cdata.mtrees:
            cdata.add_mut_lvls(lbls_key)

    use_mtypes = set()
    with open(coh_path, 'wb') as f:
        pickle.dump(cdata, f, protocol=-1)

    for use_lvls in lvls_list:
        lbls_key = ('Gene', 'Scale') + tuple(use_lvls)

        for i, (gene, muts) in enumerate(cdata.mtrees[lbls_key]):
            if len(pnt_mtype.get_samples(muts)) >= use_params['samp_cutoff']:
                gene_mtypes = {
                    mtype for mtype in muts['Point'].combtypes(
                        comb_sizes=tuple(range(
                            1, use_params['branch_combs'] + 1)),
                        min_type_size=use_params['samp_cutoff'],
                        min_branch_size=use_params['min_branch']
                        )
                    if (use_params['samp_cutoff']
                        <= len(mtype.get_samples(muts))
                        <= (n_samps - use_params['samp_cutoff']))
                    }

                # remove subgroupings that contain all of the gene's mutations
                gene_mtypes -= {mtype for mtype in gene_mtypes
                                if (len(mtype.get_samples(muts))
                                    == len(pnt_mtype.get_samples(muts)))}

                # remove subgroupings that have only one child subgrouping
                # containing all of their samples
                gene_mtypes -= {
                    mtype1
                    for mtype1, mtype2 in product(gene_mtypes, repeat=2)
                    if mtype1 != mtype2 and mtype1.is_supertype(mtype2)
                    and mtype1.get_samples(muts) == mtype2.get_samples(muts)
                    }

                gene_mtypes -= {mtype for mtype in gene_mtypes
                                if (len(mtype.get_samples(muts))
                                    == len(muts.get_samples()))}

                if i == 0:
                    gene_mtypes |= {pnt_mtype}

                use_mtypes |= {MuType({('Gene', gene): mtype})
                               for mtype in gene_mtypes}

    with open(os.path.join(out_path, "muts-list.p"), 'wb') as f:
        pickle.dump(sorted(use_mtypes), f, protocol=-1)
    with open(os.path.join(out_path, "muts-count.txt"), 'w') as fl:
        fl.write(str(len(use_mtypes)))


if __name__ == '__main__':
    main()

