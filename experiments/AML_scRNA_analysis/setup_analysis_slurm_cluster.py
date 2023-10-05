
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

from mpi4py import MPI

def main():
    parser = argparse.ArgumentParser(
        'setup_analysis',
        description="Load datasets and enumerate subgroupings to be tested."
        )

    parser.add_argument('search_params', type=str,)
    parser.add_argument('mut_lvls', type=str,)
    parser.add_argument('out_dir', type=str,)

    # parse command line arguments, create directory for enumeration output
    args = parser.parse_args()
    out_path = os.path.join(args.out_dir, 'setup')
    print('out_path variable from setup_analysis.py script: ' + str(out_path))

    # get the parameters determining which subgroupings we will look for
    lvl_lists = [('Gene', ) + lvl_list
                 for lvl_list in mut_lvls[args.mut_lvls]]
    search_dict = params[args.search_params]

    # load beatAML expression and mutation datasets
    
    # The code below uses the "beatAML" option for "get_cohort_data()" function
    # I've edited the code with an if/else statement for if beatAML is the dataset
    # or if "beatAMLwvs1to4" is the data to use.
    '''
    cdata = get_cohort_data('beatAML', 'toil__gns', lvl_lists,
                            vep_cache_dir, out_path, use_copies=False)
    '''
    cdata = get_cohort_data('beatAMLwvs1to4', 'toil__gns', lvl_lists,
                            vep_cache_dir, out_path, use_copies=False)
    
    print('cdata variable: ' + str(cdata))
    
    with bz2.BZ2File(os.path.join(out_path, "cohort-data.p.gz"), 'w') as f:
        pickle.dump(cdata, f, protocol=-1)

    # load single-cell expression data; figure out which expression features
    # overlap with those available for beatAML
    sc_expr = load_scRNA_expr()
    use_feats = set(cdata.get_features()) & set(sc_expr.columns)
    with open(os.path.join(out_path, "feat-list.p"), 'wb') as f:
        pickle.dump(use_feats, f, protocol=-1)

    total_samps = len(cdata.get_samples())
    max_samps = total_samps - search_dict['samp_cutoff']

    # intialize list of enumerated subgroupings; get list of genes mutated in
    # beatAML for which annotation info is available
    test_mtypes = set()
    test_genes = {gene for gene, _ in tuple(cdata.mtrees.values())[0]
                  if gene in cdata.gene_annot}

    # for each mutated gene, count how many samples have its point mutations
    for gene in test_genes:
        
        ##### Added by Andrew on 10/2/2023 to make sure the script is enumerating genes properly #####
        print('Current gene in test_genes variable: ' + str(gene))
        
        pnt_count = {len(cdata.mtrees[lvls][gene].get_samples())
                     for lvls in lvl_lists}

        assert len(pnt_count) == 1, (
            "Mismatching mutation trees for gene {}!".format(gene))
        pnt_count = tuple(pnt_count)[0]

        # only enumerate subgroupings if the total number of point mutants
        # exceeds the minimum number for enumerated subgroupings
        if pnt_count >= search_dict['samp_cutoff']:
            samp_dict = {None: cdata.mtrees[lvl_lists[0]][gene].get_samples()}
            gene_types = set()

            # for each set of mutation annotation levels, get the subgroupings
            # matching the search criteria
            for lvls in lvl_lists:
                
                ##### Added by Andrew on 10/2/2023 to make sure the script is enumerating properly #####
                print('Current lvls in lvl_lists variable: ' + str(lvls))
                
                use_mtree = cdata.mtrees[lvls][gene]
                
                ##### Added by Andrew on 10/2/2023 to make sure the script is enumerating properly #####
                print('Current mtree in lvls in lvl_lists for loop (use_mtree variable): ' + str(use_mtree))

                lvl_types = {
                    mtype for mtype in use_mtree.combtypes(
                        comb_sizes=tuple(
                            range(1, search_dict['branch_combs'] + 1)),
                        min_type_size=search_dict['samp_cutoff'],
                        min_branch_size=search_dict['min_branch']
                        )
                    }

                # Decided not to print this variable as it's super long - looks like:
                # {Pfam-domain IS PF00145 WITH Consequence IS missense_variant OR Pfam-domain IS PF00855 OR Pfam-domain 
                # IS none WITH Consequence IS stop_gained, etc. etc.
                #print('lvl_types variable from setup_analysis.py script: ' + str(lvl_types))
                
                # get the samples mutated for each putative subgrouping
                samp_dict.update({mtype: mtype.get_samples(use_mtree)
                                  for mtype in lvl_types})

                # filter out subgroupings with too many mutants and those
                # which contain all of the gene's point mutations
                gene_types |= {mtype for mtype in lvl_types
                               if (len(samp_dict[mtype]) <= max_samps
                                   and len(samp_dict[mtype]) < pnt_count)}

            # Get the SLURM_ARRAY_TASK_ID
            task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))

            sorted_gene_types = sorted(gene_types)
            total_len = len(sorted_gene_types)

            # Number of tasks (processes) is set by the array job size in your SLURM script (see next step)
            num_tasks = int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 1))

            tasks_per_process = total_len // num_tasks
            start_task = task_id * tasks_per_process
            end_task = (task_id + 1) * tasks_per_process if task_id != num_tasks - 1 else total_len

            # remove duplicate subgroupings, i.e. those that have the same set
            # of mutated samples as another subgrouping
            rmv_mtypes = set()
            for task_num in range(start_task, end_task):
                rmv_mtype = sorted_gene_types[task_num]
                print('rmv_mtype from sorted(gene_types) for loop in setup_analysis.py: ' + str(rmv_mtype))
                print('Current iteration of rmv_mtype for loop: ' + str(task_num) + '/' + str(total_len))
    
                rmv_lvls = rmv_mtype.get_levels()

                for cmp_mtype in sorted(set(sorted_gene_types) - {rmv_mtype} - rmv_mtypes):
                    cmp_lvls = cmp_mtype.get_levels()

                    if (samp_dict[rmv_mtype] == samp_dict[cmp_mtype]
                            and (rmv_mtype.is_supertype(cmp_mtype)
                                 or (any('domain' in lvl for lvl in rmv_lvls)
                                     and all('domain' not in lvl for lvl in cmp_lvls))
                                 or len(rmv_lvls) > len(cmp_lvls)
                                 or rmv_mtype > cmp_mtype)):
                        rmv_mtypes |= {rmv_mtype}
                        break

            #current_iteration += 1

            # update list of subgroupings to be enumerated
            test_mtypes |= {MuType({('Gene', gene): mtype})
                            for mtype in gene_types - rmv_mtypes}
            test_mtypes |= {MuType({('Gene', gene): None})}
            
            print('Updated test_mtypes variable from end of setup_analysis.py script for loop: ' + str(test_mtypes))

    ##### Added by Andrew 10/2/2023 to check what's happening with the output files below #####
    print('muts-list.p output variable sorted(test_mtypes): ' + str(sorted(test_mtypes)))
    print('muts-count.txt output variable str(len(test_mtypes)): ' + str(len(test_mtypes)))
            
    with open(os.path.join(out_path, "muts-list.p"), 'wb') as f:
        pickle.dump(sorted(test_mtypes), f, protocol=-1)
    with open(os.path.join(out_path, "muts-count.txt"), 'w') as fl:
        fl.write(str(len(test_mtypes)))
        
    print('Yay! We got to the end of the setup_analysis.py script successfully!')


if __name__ == '__main__':
    main()

