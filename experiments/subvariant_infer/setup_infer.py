
"""Enumerating the subtypes of a gene in a cohort to be isolated.

"""

import os
import sys

sys.path.extend([os.path.join(os.path.dirname(__file__), '../../..')])
if 'BASEDIR' in os.environ:
    base_dir = os.environ['BASEDIR']
else:
    base_dir = os.path.dirname(__file__)

from HetMan.experiments.subvariant_infer import *
from HetMan.experiments.utilities.load_input import load_firehose_cohort
from dryadic.features.mutations import MuType, MutComb

import argparse
from functools import reduce
from operator import or_
from itertools import combinations as combn
import dill as pickle


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
        if not isinstance(other, Mcomb):
            return NotImplemented

        return self.mtypes < other.mtypes


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
        if not isinstance(other, ExMcomb):
            return NotImplemented

        if self.all_mtype != other.all_mtype:
            raise ValueError("Cannot compare combinations from different "
                             "mutation cohorts!")

        return self.mtypes < other.mtypes

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
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='turns on diagnostic messages')

    # parse command line arguments
    args = parser.parse_args()
    use_lvls = args.mut_levels.split('__')

    # create directory where found subtypes will be stored, load cohort
    # expression and mutation data
    out_path = os.path.join(base_dir, 'setup', args.cohort, args.gene)
    os.makedirs(out_path, exist_ok=True)
    cdata = load_firehose_cohort(args.cohort, [args.gene], use_lvls)

    with open(os.path.join(out_path, 'cohort_data__levels_{}.p'.format(
            args.mut_levels)), 'wb') as f:
        pickle.dump(cdata, f)

    if args.verbose:
        print("Looking for combinations of subtypes of mutations in gene {} "
              "present in at least {} of the samples in TCGA cohort {} at "
              "annotation levels {}.\n".format(
                  args.gene, args.samp_cutoff, args.cohort, use_lvls)
             )

    use_mtypes = {
        mtype for mtype in (
            cdata.train_mut.branchtypes(min_size=args.samp_cutoff)
            - {MuType({('Scale', 'Copy'): None})}
            | {MuType({('Scale', 'Copy'): {
                ('Copy', ('DeepGain', 'ShalGain')): None}})}
            | {MuType({('Scale', 'Copy'): {
                ('Copy', ('DeepDel', 'ShalDel')): None}})}
            )
        if (args.samp_cutoff <= len(mtype.get_samples(cdata.train_mut))
            <= (len(cdata.samples) - args.samp_cutoff))
        }

    use_mcombs = {ExMcomb(cdata.train_mut, mtype) for mtype in use_mtypes}
    use_pairs = {(mtype1, mtype2) for mtype1, mtype2 in combn(use_mtypes, 2)
                 if (mtype1 & mtype2).is_empty()}
    use_mcombs |= {Mcomb(*pair) for pair in use_pairs}
    use_mcombs |= {ExMcomb(cdata.train_mut, *pair) for pair in use_pairs}

    use_mtypes |= {mcomb for mcomb in use_mcombs
                   if (args.samp_cutoff
                       <= len(mcomb.get_samples(cdata.train_mut))
                       <= (len(cdata.samples) - args.samp_cutoff))}

    out_mtypes = sorted(mtype for mtype in use_mtypes
                        if isinstance(mtype, MuType))
    out_mtypes += sorted(mtype for mtype in use_mtypes
                         if isinstance(mtype, Mcomb))
    out_mtypes += sorted(mtype for mtype in use_mtypes
                         if isinstance(mtype, ExMcomb))

    # save the list of found non-duplicate subtypes to file
    pickle.dump(
        out_mtypes,
        open(os.path.join(out_path,
                          'mtypes_list__samps_{}__levels_{}.p'.format(
                              args.samp_cutoff, args.mut_levels)),
             'wb')
        )

    # save the number of found subtypes to file
    with open(os.path.join(out_path,
                           'mtypes_count__samps_{}__levels_{}.txt'.format(
                               args.samp_cutoff, args.mut_levels)),
              'w') as fl:

        fl.write(str(len(out_mtypes)))


if __name__ == '__main__':
    main()

