
"""Enumerating the subtypes of a gene in a cohort to be isolated.

"""

import os
import sys

sys.path.extend([os.path.join(os.path.dirname(__file__), '../../..')])
if 'BASEDIR' in os.environ:
    base_dir = os.environ['BASEDIR']
else:
    base_dir = os.path.dirname(__file__)

from HetMan.experiments.subvariant_isolate import *
from HetMan.experiments.utilities.load_input import load_firehose_cohort
from dryadic.features.mutations import MuType, MutComb

import argparse
from functools import reduce
from operator import or_
from itertools import combinations as combn
import dill as pickle


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

    if args.verbose:
        print("Looking for combinations of subtypes of mutations in gene {} "
              "present in at least {} of the samples in TCGA cohort {} at "
              "annotation levels {}.\n".format(
                  args.gene, args.samp_cutoff, args.cohort, use_lvls)
             )

    # find combinations of up to two point mutation subtypes present in enough
    # samples in the cohort to meet the frequency cutoff criteria
    pnt_mtypes = cdata.train_mut['Point'].find_unique_subtypes(
        max_types=1000, max_combs=3, verbose=2,
        sub_levels=use_lvls, min_type_size=args.samp_cutoff
        )

    # filter out the subtypes that appear in too many samples for there to
    # be a wild-type class of sufficient size for classification
    pnt_mtypes = {MuType({('Scale', 'Point'): mtype}) for mtype in pnt_mtypes
                  if (len(mtype.get_samples(cdata.train_mut['Point']))
                      <= (len(cdata.samples) - args.samp_cutoff))}
    pnt_mtypes |= {MuType({('Scale', 'Point'): None})}

    # find copy number alterations whose frequency falls within the cutoffs
    cna_mtypes = cdata.train_mut['Copy'].branchtypes(
        min_size=args.samp_cutoff)
    cna_mtypes |= {MuType({('Copy', ('HetGain', 'HomGain')): None})}
    cna_mtypes |= {MuType({('Copy', ('HetDel', 'HomDel')): None})}

    cna_mtypes = {MuType({('Scale', 'Copy'): mtype}) for mtype in cna_mtypes
                  if (len(mtype.get_samples(cdata.train_mut['Copy']))
                      <= (len(cdata.samples) - args.samp_cutoff))}

    # get the mutation type corresponding to the union of all mutations
    # present in the cohort, consolidate the subtypes found thus far

    # for each subtype, check if it is present in enough samples even after we
    # remove the samples with another mutation or alteration present
    use_mtypes = pnt_mtypes | cna_mtypes
    only_mcombs = {ExMcomb(cdata.train_mut, mtype) for mtype in use_mtypes}
    pair_mcombs = {ExMcomb(cdata.train_mut, mtype1, mtype2)
                   for mtype1, mtype2 in combn(use_mtypes, 2)
                   if (mtype1 & mtype2).is_empty()}

    use_mcombs = {
        mcomb for mcomb in only_mcombs | pair_mcombs
        if len(mcomb.get_samples(cdata.train_mut)) >= args.samp_cutoff
        }

    if args.verbose:
        print("\nFound {} exclusive sub-types and {} combination sub-types "
              "to isolate!".format(
                  len([x for x in use_mcombs if len(x.mtypes) == 1]),
                  len([x for x in use_mcombs if len(x.mtypes) == 2])
                ))

    # save the list of found non-duplicate subtypes to file
    pickle.dump(
        sorted(use_mcombs),
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

        fl.write(str(len(use_mcombs)))

    with open(os.path.join(out_path, 'cohort_data__levels_{}.p'.format(
            args.mut_levels)), 'wb') as f:
        pickle.dump(cdata, f)


if __name__ == '__main__':
    main()

