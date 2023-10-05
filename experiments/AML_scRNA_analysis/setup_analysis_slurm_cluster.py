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
from itertools import product
import time

def main():
    parser = argparse.ArgumentParser(
        'setup_analysis',
        description="Load datasets and enumerate subgroupings to be tested."
    )

    parser.add_argument('search_params', type=str)
    parser.add_argument('mut_lvls', type=str)
    parser.add_argument('out_dir', type=str)

    args = parser.parse_args()
    out_path = os.path.join(args.out_dir, 'setup')
    print('out_path variable from setup_analysis.py script: ' + str(out_path))

    lvl_lists = [('Gene', ) + lvl_list for lvl_list in mut_lvls[args.mut_lvls]]
    search_dict = params[args.search_params]

    cdata = get_cohort_data('beatAMLwvs1to4', 'toil__gns', lvl_lists, vep_cache_dir, out_path, use_copies=False)
    print('cdata variable: ' + str(cdata))
    
    with bz2.BZ2File(os.path.join(out_path, "cohort-data.p.gz"), 'w') as f:
        pickle.dump(cdata, f, protocol=-1)

    sc_expr = load_scRNA_expr()
    use_feats = set(cdata.get_features()) & set(sc_expr.columns)
    with open(os.path.join(out_path, "feat-list.p"), 'wb') as f:
        pickle.dump(use_feats, f, protocol=-1)

    total_samps = len(cdata.get_samples())
    max_samps = total_samps - search_dict['samp_cutoff']

    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    num_tasks = int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 1))

    local_test_mtypes = set()
    test_genes = {gene for gene, _ in tuple(cdata.mtrees.values())[0] if gene in cdata.gene_annot}

    for gene in test_genes:
        pnt_count = {len(cdata.mtrees[lvls][gene].get_samples()) for lvls in lvl_lists}
        assert len(pnt_count) == 1, "Mismatching mutation trees for gene {}!".format(gene)
        pnt_count = tuple(pnt_count)[0]

        if pnt_count >= search_dict['samp_cutoff']:
            samp_dict = {None: cdata.mtrees[lvl_lists[0]][gene].get_samples()}
            gene_types = set()

            for lvls in lvl_lists:
                print('Current lvls in lvl_lists variable: ' + str(lvls))
                use_mtree = cdata.mtrees[lvls][gene]
                print('Current mtree in lvls in lvl_lists for loop (use_mtree variable): ' + str(use_mtree))

                lvl_types = {
                    mtype for mtype in use_mtree.combtypes(
                        comb_sizes=tuple(range(1, search_dict['branch_combs'] + 1)),
                        min_type_size=search_dict['samp_cutoff'],
                        min_branch_size=search_dict['min_branch']
                    )
                }

                samp_dict.update({mtype: mtype.get_samples(use_mtree) for mtype in lvl_types})
                gene_types |= {mtype for mtype in lvl_types if (len(samp_dict[mtype]) <= max_samps and len(samp_dict[mtype]) < pnt_count)}

            local_test_mtypes |= {MuType({('Gene', gene): mtype}) for mtype in gene_types}

    # Each node writes its results to a unique temporary file
    temp_filename = os.path.join(out_path, f"test_mtypes_temp_{task_id}.p")
    with open(temp_filename, 'wb') as f:
        pickle.dump(local_test_mtypes, f, protocol=-1)

    # For non-master nodes, just write a done flag and then finish.
    if task_id != 0:
        with open(os.path.join(out_path, f"done_{task_id}.flag"), 'w') as flag_file:
            flag_file.write('done')
    else:
        # Master node waits for other nodes to finish and aggregates their results
        all_tasks_done = False
        while not all_tasks_done:
            all_tasks_done = all(os.path.exists(os.path.join(out_path, f"done_{i}.flag")) for i in range(1, num_tasks))
            if not all_tasks_done:
                time.sleep(10)

        aggregated_test_mtypes = set()
        for i in range(num_tasks):  # Include the master node's results
            temp_filename = os.path.join(out_path, f"test_mtypes_temp_{i}.p")
            with open(temp_filename, 'rb') as f:
                node_test_mtypes = pickle.load(f)
                aggregated_test_mtypes.update(node_test_mtypes)
            os.remove(temp_filename)
            if i != 0:  # Master node flag should not be removed yet, as its task is not done
                os.remove(os.path.join(out_path, f"done_{i}.flag"))

        # Writing final aggregated results to the output files
        with open(os.path.join(out_path, "muts-list.p"), 'wb') as f:
            pickle.dump(sorted(aggregated_test_mtypes), f, protocol=-1)
        with open(os.path.join(out_path, "muts-count.txt"), 'w') as fl:
            fl.write(str(len(aggregated_test_mtypes)))

        print('Master node: Finished aggregating and writing results!')

    print(f'Finished task {task_id}')

if __name__ == '__main__':
    main()

