
import sys, os
sys.path.extend(['/home/exacloud/lustre1/CompBio/mgrzad/bergamot/'])

from HetMan.features.expression import get_expr_bmeg
from HetMan.features.variants import get_variants_mc3
from HetMan.features.cohorts import VariantCohort

import numpy as np
import synapseclient
import pickle
from itertools import combinations as combn


def main(argv):
    """Runs the experiment."""

    print(argv)
    out_path = os.path.join(os.path.dirname(__file__), 'output', argv[0])
    coh_lbl = 'TCGA-{}'.format(argv[0])

    syn = synapseclient.Synapse()
    syn.login()
    mc3_data = get_variants_mc3(syn)
    expr_data = get_expr_bmeg(coh_lbl)
    print(expr_data.shape)

    freq_cutoff = 20
    brca_mc3 = mc3_data.ix[mc3_data['Sample'].isin(expr_data.index), :]
    gene_counts = brca_mc3.groupby(by='Gene').count()['Sample']
    common_genes = set(gene_counts[gene_counts >= freq_cutoff].index)
    print(len(common_genes))

    cdata = VariantCohort(syn, cohort=coh_lbl, mut_genes=common_genes,
                          mut_levels=['Gene', 'Type', 'Location'],
                          cv_seed=99)

    sub_mtypes = cdata.train_mut.subtypes(
        min_size=int(freq_cutoff * 2.0 / 3.0)
        )
    print(len(sub_mtypes))

    mutex_cutoff = 1 / 3.0
    mutex_dict = {}

    for mtype1, mtype2 in combn(sub_mtypes, 2):
        stat1 = cdata.test_pheno(mtype1)
        stat2 = cdata.test_pheno(mtype2)
        
        if np.sum(stat1 & ~stat2) >= 5 and np.sum(~stat1 & stat2) >= 5:
            mutex_val = cdata.mutex_test(mtype1, mtype2)
                                            
            if mutex_val < mutex_cutoff:
                mutex_dict[(mtype1, mtype2)] = cdata.mutex_test(mtype1, mtype2)

    print(len(mutex_dict))
    pickle.dump(list(mutex_dict.items()), open(out_path + '/mutex_dict.p', 'wb'))


if __name__ == "__main__":
    main(sys.argv[1:])

