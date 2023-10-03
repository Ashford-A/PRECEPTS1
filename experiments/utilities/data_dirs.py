
from .data_locs import (
    firehose_dir, syn_root, metabric_dir, baml_dir, ccle_dir, gencode_dir,
    oncogene_list, subtype_file, vep_cache_dir, domain_dir, expr_sources
    )

import pipes
import argparse


def choose_source(cohort):
    # choose the source of expression data to use for this tumour cohort
    coh_base = cohort.split('_')[0]

    if coh_base in ['beatAML', 'beatAMLwvs1to4']:
        use_src = 'toil__gns'
    elif coh_base in ['METABRIC', 'CCLE']:
        use_src = 'microarray'

    # default to using Broad Firehose expression calls for TCGA cohorts
    else:
        use_src = 'Firehose'

    return use_src


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('cohort')
    parser.add_argument('--expr_source')
    args = parser.parse_args()

    if args.expr_source is None:
        expr_source = choose_source(args.cohort)
    else:
        expr_source = args.expr_source

    if expr_source == 'microarray':
        if args.cohort.split('_')[0] == 'METABRIC':
            coh_dir = metabric_dir

        else:
            raise ValueError("Unrecognized microarray cohort "
                             "`{}`!".format(args.expr_source))

    elif expr_source == 'Firehose':
        coh_dir = firehose_dir

    elif expr_source == 'toil__gns':
        if args.cohort.split('_')[0] == 'beatAML':
            coh_dir = baml_dir
        elif args.cohort.split('_')[0] == 'beatAMLwvs1to4':
            coh_dir = baml_dir
        else:
            coh_dir = expr_sources['toil']

    else:
        raise ValueError("Unrecognized source of cohort data "
                         "`{}`!".format(args.expr_source))

    print("export COH_DIR=%s" % (pipes.quote(coh_dir)))
    print("export GENCODE_DIR=%s" % (pipes.quote(gencode_dir)))
    print("export ONCOGENE_LIST=%s" % (pipes.quote(oncogene_list)))
    print("export SUBTYPE_LIST=%s" % (pipes.quote(subtype_file)))


if __name__ == '__main__':
    main()
