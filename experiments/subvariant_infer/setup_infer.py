
import os
import sys
base_dir = os.path.dirname(__file__)
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.experiments.subvariant_infer import *
from HetMan.experiments.utilities.load_input import load_firehose_cohort
from HetMan.features.cohorts.beatAML import BeatAmlCohort
from dryadic.features.mutations import MuType, MutComb

import argparse
import pandas as pd
import dill as pickle

from functools import reduce
from operator import or_
from itertools import combinations as combn
from itertools import product


class Mcomb(MutComb):

    def __new__(cls, *mtypes):
        return super().__new__(cls, *mtypes)

    def __hash__(self):
        value = 0x230199 ^ len(self.mtypes)
        value += hash(self.mtypes)

        if value == -1:
            value = -2

        return value

    def __getnewargs__(self):
        return tuple(self.mtypes)

    def __str__(self):
        return ' & '.join(str(mtype) for mtype in sorted(self.mtypes))

    def __repr__(self):
        return 'BOTH {}'.format(
            ' AND '.join(repr(mtype) for mtype in sorted(self.mtypes)))

    def __eq__(self, other):
        if not isinstance(other, Mcomb):
            eq = False
        else:
            eq = self.mtypes == other.mtypes

        return eq

    def __lt__(self, other):
        if isinstance(other, Mcomb):
            lt = self.mtypes < other.mtypes

        elif isinstance(other, MuType):
            lt = False
        elif isinstance(other, ExMcomb):
            lt = True
        else:
            lt = NotImplemented

        return lt


class ExMcomb(MutComb):

    def __new__(cls, mtree, *mtypes):
        if isinstance(mtree, MuType):
            all_mtype = mtree
        else:
            all_mtype = MuType(mtree.allkey())

        obj = super().__new__(cls, *mtypes,
                              not_mtype=all_mtype - reduce(or_, mtypes))
        obj.all_mtype = all_mtype
        obj.cur_level = all_mtype.cur_level

        return obj

    def __hash__(self):
        value = 0x981324 ^ (len(self.mtypes) * hash(self.all_mtype))
        value += hash(self.mtypes)

        if value == -1:
            value = -2

        return value

    def __getnewargs__(self):
        return (self.all_mtype,) + tuple(self.mtypes)

    def __str__(self):
        return ' & '.join(str(mtype) for mtype in sorted(self.mtypes))

    def __repr__(self):
        return 'ONLY {}'.format(
            ' AND '.join(repr(mtype) for mtype in sorted(self.mtypes)))

    def __eq__(self, other):
        if not isinstance(other, ExMcomb):
            eq = False

        else:
            eq = self.all_mtype == other.all_mtype
            eq &= self.mtypes == other.mtypes

        return eq

    def __lt__(self, other):
        if isinstance(other, ExMcomb):
            if self.all_mtype != other.all_mtype:
                raise ValueError("Cannot compare combinations from different "
                                 "mutation cohorts!")
 
            lt = self.mtypes < other.mtypes

        elif isinstance(other, (MuType, Mcomb)):
            lt = False
        else:
            lt = NotImplemented

        return lt

    def get_sorted_levels(self):
        return self.all_mtype.get_sorted_levels()


def main():
    parser = argparse.ArgumentParser(
        "Set up the gene subtype expression effect isolation experiment by "
        "enumerating the subtypes to be tested."
        )

    # create positional command line arguments
    parser.add_argument('cohort', type=str, help="which TCGA cohort to use")
    parser.add_argument('gene', type=str, help="which gene to consider")
    parser.add_argument('mut_levels', type=str,
                        help="the mutation property levels to consider")

    # create optional command line arguments
    parser.add_argument('--samp_cutoff', type=int, default=20,
                        help='subtype sample frequency threshold')
    parser.add_argument('--setup_dir', type=str, default=base_dir)

    # parse command line arguments
    args = parser.parse_args()
    out_path = os.path.join(args.setup_dir, 'setup')
    use_lvls = args.mut_levels.split('__')

    if args.cohort == 'beatAML':
        cdata = BeatAmlCohort(use_lvls, [args.gene], expr_source='toil__gns',
                              expr_file=beatAML_files['expr'],
                              samp_file=beatAML_files['samps'], syn=syn,
                              annot_file=annot_file, cv_seed=709, test_prop=0)

    else:
        cdata = load_firehose_cohort(args.cohort, [args.gene], use_lvls,
                                     cv_seed=709, test_prop=0)

    with open(os.path.join(out_path, "cohort-data.p"), 'wb') as f:
        pickle.dump(cdata, f)

    use_mtypes = {
        mtype for mtype in (
            cdata.mtree.branchtypes(min_size=args.samp_cutoff)
            - {MuType({('Scale', 'Copy'): None})}
            | {MuType({('Scale', 'Copy'): {
                ('Copy', ('DeepGain', 'ShalGain')): None}})}
            | {MuType({('Scale', 'Copy'): {
                ('Copy', ('DeepDel', 'ShalDel')): None}})}
            )
        if (args.samp_cutoff <= len(mtype.get_samples(cdata.mtree))
            <= (len(cdata.get_samples()) - args.samp_cutoff))
        }

    if args.mut_levels != 'Location__Protein':
        use_mtypes -= {MuType({('Scale', 'Point'): None})}

    use_mtypes -= {mtype1 for mtype1, mtype2 in product(use_mtypes, repeat=2)
                   if mtype1 != mtype2 and mtype1.is_supertype(mtype2)
                   and (mtype1.get_samples(cdata.mtree)
                        == mtype2.get_samples(cdata.mtree))}

    use_pairs = {(mtype1, mtype2) for mtype1, mtype2 in combn(use_mtypes, 2)
                 if (mtype1 & mtype2).is_empty()}
    use_mcombs = {Mcomb(*pair) for pair in use_pairs}
    use_mcombs |= {ExMcomb(cdata.mtree, *pair) for pair in use_pairs}

    if args.mut_levels != 'Location__Protein':
        use_mtypes = {
            mtype for mtype in use_mtypes
            if (mtype & MuType({('Scale', 'Copy'): None})).is_empty()
            }

    use_mcombs |= {ExMcomb(cdata.mtree, mtype) for mtype in use_mtypes}
    use_mtypes |= {mcomb for mcomb in use_mcombs
                   if (args.samp_cutoff
                       <= len(mcomb.get_samples(cdata.mtree))
                       <= (len(cdata.get_samples()) - args.samp_cutoff))}

    with open(os.path.join(out_path, "muts-list.p"), 'wb') as f:
        pickle.dump(sorted(use_mtypes), f)
    with open(os.path.join(out_path, "muts-count.txt"), 'w') as fl:
        fl.write(str(len(use_mtypes)))


if __name__ == '__main__':
    main()

