
from ..utilities.data_dirs import (firehose_dir, syn_root, metabric_dir,
                                   baml_dir, gencode_dir, oncogene_list,
                                   subtype_file, vep_cache_dir, expr_sources)

import pipes
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('expr_source')
    parser.add_argument('cohort')
    args = parser.parse_args()

    if args.expr_source == 'microarray':
        if args.cohort.split('_')[0] == 'METABRIC':
            coh_dir = metabric_dir

        else:
            raise ValueError("Unrecognized microarray cohort "
                             "`{}`!".format(args.expr_source))

    elif args.expr_source == 'Firehose':
        coh_dir = firehose_dir

    elif args.expr_source == 'toil__gns':
        if args.cohort.split('_')[0] == 'beatAML':
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
