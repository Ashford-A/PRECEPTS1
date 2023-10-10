from ..utilities.mutations import pnt_mtype
from dryadic.features.mutations import MuType
from .param_list import params, mut_lvls
from .utils import load_scRNA_expr
from ..utilities.data_dirs import vep_cache_dir, expr_sources
from ...features.cohorts.utils import get_cohort_data

import os
import argparse
import bz2
import dill as pickle

def preprocess(args):
    out_path = os.path.join(args.out_dir, 'setup')
    print('Preprocessing and saving data to: ' + str(out_path))

    lvl_lists = [('Gene', ) + lvl_list for lvl_list in mut_lvls[args.mut_lvls]]
    search_dict = params[args.search_params]

    cdata = get_cohort_data('beatAMLwvs1to4', 'toil__gns', lvl_lists, vep_cache_dir, out_path, use_copies=False)
    
    with bz2.BZ2File(os.path.join(out_path, "cohort-data.p.gz"), 'w') as f:
        pickle.dump(cdata, f, protocol=-1)

    sc_expr = load_scRNA_expr()
    use_feats = set(cdata.get_features()) & set(sc_expr.columns)
    with open(os.path.join(out_path, "feat-list.p"), 'wb') as f:
        pickle.dump(use_feats, f, protocol=-1)

    print('Preprocessing finished.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'preprocess_script',
        description="Load datasets and save them for subsequent analysis."
    )

    parser.add_argument('search_params', type=str)
    parser.add_argument('mut_lvls', type=str)
    parser.add_argument('out_dir', type=str)

    args = parser.parse_args()
    preprocess(args)

