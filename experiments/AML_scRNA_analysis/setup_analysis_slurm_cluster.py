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
import time

import hashlib 


def process_gene(cdata, gene, lvl_lists, search_dict, max_samps):
    # Extract common calculations outside the loop
    pnt_count = len(cdata.mtrees[lvl_lists[0]][gene].get_samples())
    samp_dict = {None: cdata.mtrees[lvl_lists[0]][gene].get_samples()}
    gene_types = set()

    for lvls in lvl_lists:
        use_mtree = cdata.mtrees[lvls][gene]
        lvl_types = {
            mtype for mtype in use_mtree.combtypes(
                comb_sizes=tuple(range(1, search_dict['branch_combs'] + 1)),
                min_type_size=search_dict['samp_cutoff'],
                min_branch_size=search_dict['min_branch']
            )
        }
        samp_dict.update({mtype: mtype.get_samples(use_mtree) for mtype in lvl_types})
        gene_types |= {mtype for mtype in lvl_types if (len(samp_dict[mtype]) <= max_samps and len(samp_dict[mtype]) < pnt_count)}

    # Remove duplicate subgroupings
    rmv_mtypes = set()
    for rmv_mtype in sorted(gene_types):
        rmv_lvls = rmv_mtype.get_levels()
        for cmp_mtype in sorted(gene_types - {rmv_mtype} - rmv_mtypes):
            cmp_lvls = cmp_mtype.get_levels()

            if (samp_dict[rmv_mtype] == samp_dict[cmp_mtype]
                    and (rmv_mtype.is_supertype(cmp_mtype)
                        or (any('domain' in lvl for lvl in rmv_lvls)
                            and all('domain' not in lvl for lvl in cmp_lvls))
                        or len(rmv_lvls) > len(cmp_lvls)
                        or rmv_mtype > cmp_mtype)):
                rmv_mtypes |= {rmv_mtype}
                break

    # Update the local_test_mtypes set
    local_test_mtypes = {MuType({('Gene', gene): mtype})
                          for mtype in gene_types - rmv_mtypes}
    local_test_mtypes.add(MuType({('Gene', gene): None}))

    return local_test_mtypes


def calculate_checksum(filename):
    """Calculate the SHA256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(filename, "rb") as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def save_checksum(filename, checksum):
    """Save the checksum to a file."""
    with open(f"{filename}.sha256", "w") as f:
        f.write(checksum)


def verify_checksum(filename):
    """Verify the checksum of a file against the saved checksum."""
    saved_checksum = ""
    with open(f"{filename}.sha256", "r") as f:
        saved_checksum = f.read().strip()
    current_checksum = calculate_checksum(filename)
    return saved_checksum == current_checksum


def gene_already_processed(out_path, task_id):
    """Check if the results for a specific task_id/gene already exist and are intact."""
    temp_filename = os.path.join(out_path, f"test_mtypes_temp_{task_id}.p")
    if not os.path.exists(temp_filename):
        return False
    
    # Verify checksum
    if not verify_checksum(temp_filename):
        print(f"Checksum verification failed for task {task_id}. Re-processing.")
        return False

    # You can add more content verifications here, if needed.
    
    return True


def main():
    parser = argparse.ArgumentParser(
        'parallel_script',
        description="Distributed analysis based on preprocessed data."
    )

    parser.add_argument('search_params', type=str)
    parser.add_argument('mut_lvls', type=str)
    parser.add_argument('out_dir', type=str)

    args = parser.parse_args()
    out_path = os.path.join(args.out_dir, 'setup')
    
    lvl_lists = [('Gene', ) + lvl_list for lvl_list in mut_lvls[args.mut_lvls]]
    search_dict = params[args.search_params]
    cdata = get_cohort_data('beatAMLwvs1to4', 'toil__gns', lvl_lists, vep_cache_dir, out_path, use_copies=False)
    sc_expr = load_scRNA_expr()
    use_feats = set(cdata.get_features()) & set(sc_expr.columns)
    
    total_samps = len(cdata.get_samples())
    max_samps = total_samps - search_dict['samp_cutoff']

    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    num_tasks = int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 1)) 
    
    # Check if this task/gene has already been processed and is intact
    if gene_already_processed(out_path, task_id):
        print(f"Task {task_id} has already been processed. Skipping.")
        return

    local_test_mtypes = set()
    test_genes = {gene for gene, _ in tuple(cdata.mtrees.values())[0] if gene in cdata.gene_annot}

    # Distribute genes among Slurm tasks
    gene_partition = len(test_genes) // num_tasks
    gene_start = task_id * gene_partition
    gene_end = (task_id + 1) * gene_partition if task_id < num_tasks - 1 else len(test_genes)

    for gene in list(test_genes)[gene_start:gene_end]:
        gene_results = process_gene(cdata, gene, lvl_lists, search_dict, max_samps)
        local_test_mtypes.update(gene_results)

    # Each node writes its results to a unique temporary file
    temp_filename = os.path.join(out_path, f"test_mtypes_temp_{task_id}.p")
    with open(temp_filename, 'wb') as f:
        pickle.dump(local_test_mtypes, f, protocol=-1)
    
    save_checksum(temp_filename, calculate_checksum(temp_filename))

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
                os.remove(os.path.join(out_path, f"test_mtypes_temp_{i}.p.sha256"))

        # Writing final aggregated results to the output files
        with open(os.path.join(out_path, "muts-list.p"), 'wb') as f:
            pickle.dump(sorted(aggregated_test_mtypes), f, protocol=-1)
        with open(os.path.join(out_path, "muts-count.txt"), 'w') as fl:
            fl.write(str(len(aggregated_test_mtypes)))

        print('Master node: Finished aggregating and writing results!')

        if i == 0:
            os.remove(os.path.join(out_path, f"done_{i}.flag"))
            os.remove(os.path.join(out_path, f"test_mtypes_temp_{i}.p.sha256"))

    print(f'Finished task {task_id}')


if __name__ == '__main__':
    main()

