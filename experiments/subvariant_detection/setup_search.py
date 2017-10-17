
import os
base_dir = os.path.dirname(__file__)

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.features.expression import get_expr_firehose
from HetMan.features.variants import get_variants_mc3
from HetMan.features.cohorts import VariantCohort

import numpy as np
import synapseclient
import dill as pickle

from itertools import combinations as combn
from itertools import chain

firehose_dir = '/home/exacloud/lustre1/CompBio/mgrzad/input-data/firehose'


def main(argv):
    """Runs the experiment."""

    # get the directory where results will be saved and the name of the
    # TCGA cohort we will be using
    print(argv)
    out_path = os.path.join(base_dir, 'output', argv[0], argv[1], 'search')
    freq_cutoff = int(argv[2])

    # load the expression data and the gene's mutation data
    # for the given cohort
    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = ("/home/exacloud/lustre1/CompBio/"
                                "mgrzad/input-data/synapse")
    syn.login()

    expr_data = get_expr_firehose(argv[0], firehose_dir)
    mc3_data = get_variants_mc3(syn)
    print(expr_data.shape)

    expr_mc3 = mc3_data.loc[mc3_data['Sample'].isin(expr_data.index), :]
    gene_counts = expr_mc3.groupby(by='Gene').count()['Sample']
    common_genes = set(gene_counts[gene_counts >= freq_cutoff].index)
    print(len(common_genes))

    cdata = VariantCohort(
        cohort=argv[0], mut_genes=common_genes,
        mut_levels=['Gene', 'Form_base', 'Exon', 'Protein'],
        expr_source='Firehose', data_dir=firehose_dir,
        cv_prop=1.0, syn=syn
        )

    # finds the sub-types satisfying the sample frequency criterion, starting
    # with the ones that are combinations of individual branches
    use_mtypes = set()
    use_sampsets = set()
    comb_size = 0

    while len(use_mtypes) < 8000 and comb_size <= 5:
        comb_size += 1
        sub_mtypes = cdata.train_mut.combtypes(comb_sizes=(comb_size, ),
                                               min_type_size=freq_cutoff)

        print("Found {} sub-types that are combinations of {} branch(es)."
              .format(len(sub_mtypes), comb_size))

        if len(sub_mtypes) < 7500:
            for mtype in sub_mtypes.copy():
                mtype_sampset = frozenset(mtype.get_samples(cdata.train_mut))

                if mtype_sampset in use_sampsets:
                    print("Removing functionally duplicate MuType {}"
                          .format(mtype))
                    sub_mtypes.remove(mtype)

                else:
                    use_sampsets.update(mtype_sampset)

            use_mtypes |= sub_mtypes

        else:
            break
        
    print("Using {} sub-types of branch combinations."
          .format(len(use_mtypes)))

    # save the list of sub-types to file
    pickle.dump(list(use_mtypes),
                open(os.path.join(out_path, 'tmp/mtype_list.p'), 'wb'))

if __name__ == "__main__":
    main(sys.argv[1:])

