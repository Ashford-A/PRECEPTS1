
"""Unit tests for mutation subtype representations.

This file contains unit tests for:
    MuTypes, a class for representing mutation subtypes containing
             arbitrary combinations of mutation properties
    MuTrees, a class for representing the hierarchy of mutation types present
             in a particular set of samples

See Also:
    :module:`..features.variants`: Contains the classes that are tested here.

Author: Michal Grzadkowski <grzadkow@ohsu.edu>

"""

from ..features.variants import MuType, MuTree

import numpy as np
import pandas as pd

import pytest
import pickle

from functools import reduce
from itertools import combinations as combn
from itertools import chain, product

import os
DATA_PATH = os.path.join(os.path.dirname(__file__), 'mut_objs/')


# .. objects that are re-used across multiple tests ..
@pytest.fixture(scope='module')
def mtree_tester(request):
    """Create a tree containing samples with particular mutation types."""
    return MtreeTester(request.param)


def muts_id(val):
    """Parses the name of a file containing mutation annotation tables used in
       testing to get a mutation set label."""
    if 'muts_' in val[0]:
        return val[0].split('muts_')[-1].split('.p')[0]
    else:
        return val


class MtreeTester(object):
    """Defines a mutation tree of samples for use in testing.

    Attributes:
        muts_lbl (str): The name of a mutation set in ./mut_objs.
        mut_levels (list of str): Which annotation levels to include.

    Examples:
        >>> MtreeTester(('small', ['Gene', 'Exon']))
        >>> MtreeTester(('TP53', ['Type', 'Exon', 'Location']))
        >>> MtreeTester(('large', ['Gene', 'Form', 'Protein']))

    """

    # where test mutation datasets are expected to be located
    mut_file_base = DATA_PATH + 'muts_'

    def __init__(self, request):
        self.muts_lbl = request[0]
        self.mut_levels = request[1]

    def get_muts_mtree(self):
        """Loads the MuTree defined by this instance."""

        with open(self.mut_file_base + self.muts_lbl + '.p', 'rb') as fl:
            muts = pickle.load(fl)
        mtree = MuTree(muts, levels=self.mut_levels)

        return muts, mtree, self.mut_levels


@pytest.fixture(scope='function')
def mtype_tester(request):
    """Create a set of mutation subtypes."""
    return MtypeTester(request.param)


class MtypeTester(object):
    """Defines a set of mutation subtypes for use in testing.

    Attributes:
        type_lbl (str): Corresponds to a set of subtypes present in the
                        TypeTester.mtypes object. Can be '_all' in which case
                        all subtypes are loaded together.

    Examples:
        >>> MtypeTester('small')
        >>> MtypeTester('synonym')

    """

    # .. pre-defined mutation subtype sets ..
    mtypes = {

        # .. basic subtypes present in the ./mut_objs/muts_small.p set
        'small': (
            MuType({('Gene', 'TTN'): None}),
            MuType({
                ('Gene', 'TTN'): {
                    ('Form', 'Missense_Mutation'): {
                        ('Exon', ('326/363', '10/363')): None
                        }}}),
            MuType({
                ('Gene', 'CDH1'): None,
                ('Gene', 'TTN'): {
                    ('Form', 'Missense_Mutation'): {
                        ('Exon', ('302/363', '10/363')): None
                        }}}),
            MuType({('Form', 'Silent'): None}),
            MuType({('Gene', ('CDH1', 'TTN')): None})
            ),

        # .. subtypes which should never return any samples
        'blank': (
            MuType({('Gene', ('TTN', 'LOL5')): None}),
            MuType({('Gene', 'TTN'): {('Form', 'silly'): None}}),
            MuType({('Genie', 'Alladin'): None}),
            MuType({('Gene', 'TTN'): {('Gene', 'TTN2'): None}}),
            ),

        # .. paired subtypes which represent the same mutation sets
        'synonym': (
            MuType({('Gene', ('TTN', 'TP53')): None}),
            MuType({('Gene', 'TTN'): None, ('Gene', 'TP53'): None}),
            MuType({
                ('Gene', 'TTN'): {
                    ('Form', ('Silent', 'Splice')): {
                        ('Protein', ('p43T', 'p55H')): None
                        }}}),
            MuType({
                ('Gene', 'TTN'): {
                    ('Form', 'Silent'): {
                        ('Protein', 'p43T'): None
                        },
                    ('Form', ('Splice', 'Silent')): {
                        ('Protein', 'p55H'): None
                        },
                    ('Form', 'Splice'): {
                        ('Protein', 'p43T'): None
                        }}}),
            ),

        # .. subtypes unique to mutations in the TP53 gene
        'TP53': (
            MuType({('Form', 'Splice_Site'): None}),
            MuType({('Exon', ('7/11', '8/11')): None}),
            MuType({('Protein', ('p.R158L', '.')): None}),
            ),

        # .. subtypes designed for use in testing binary operations
        'binary': (
            MuType({
                ('Gene', ('BRAF', 'TP53')): {
                    ('Form', 'Splice_Site'): None
                    },
                ('Gene', 'KRAS'): {
                    ('Form', ('Splice_Site', 'Frame_Shift')): None
                    }}),
            MuType({
                ('Gene', 'TP53'): {
                    ('Form', ('Splice_Site', 'Missense_Mutation')): None
                    }}),
            MuType({
                ('Gene', 'KRAS'): {
                    ('Form', ('Splice_Site', 'Frame_Shift')): None
                    },
                ('Gene', 'BRAF'): {
                    ('Form', 'Splice_Site'): None
                    },
                ('Gene', 'TP53'): {
                    ('Form', ('Splice_Site', 'Missense_Mutation')): None
                    }}),
            MuType({('Gene', 'TP53'): {('Form', 'Splice_Site'): None}}),
            )
        }

    def __init__(self, request):
        self.type_lbl = request

    def get_types(self):
        """Loads the mutation subtype sets defined by this instance."""

        if self.type_lbl in self.mtypes:
            return self.mtypes[self.type_lbl]

        elif self.type_lbl == '_all':
            return reduce(lambda x, y: x + y, self.mtypes.values())

        else:
            raise ValueError("Unknown MuType test set!")


@pytest.mark.parametrize('mtree_tester',
                         [('small', ('Gene', 'Form', 'Exon')),
                          ('TP53', ('Form', 'Exon', 'Protein'))],
                         ids=muts_id, indirect=True, scope="class")
class TestCaseBasicMuTree:
    """Tests for basic functionality of MuTrees."""

    def test_levels(self, mtree_tester):
        """Does the tree correctly implement nesting of mutation levels?"""
        muts, mtree, mut_lvls = mtree_tester.get_muts_mtree()
        assert set(mut_lvls) == mtree.get_levels()

    def test_keys(self, mtree_tester):
        """Does the tree correctly implement key retrieval of subtrees?"""
        muts, mtree, mut_lvls = mtree_tester.get_muts_mtree()

        if len(mtree.get_levels()) > 1:
            for vals, _ in muts.groupby(mut_lvls):
                assert mtree._child[vals[0]][vals[1:]] == mtree[vals]
                assert mtree[vals[:-1]]._child[vals[-1]] == mtree[vals]

        else:
            for val in set(muts[mtree.mut_level]):
                assert mtree._child[val] == mtree[val]

    def test_structure(self, mtree_tester):
        """Is the internal structure of the tree correct?"""
        muts, mtree, mut_lvls = mtree_tester.get_muts_mtree()

        assert set(mtree._child.keys()) == set(muts[mtree.mut_level])
        assert mtree.depth == 0

        lvl_sets = {i: mut_lvls[:i] for i in range(1, len(mut_lvls))}
        for i, lvl_set in lvl_sets.items():

            for vals, mut in muts.groupby(lvl_set):
                assert mtree[vals].depth == i
                assert (set(mtree[vals]._child.keys())
                        == set(mut[mut_lvls[i]]))

    def test_print(self, mtree_tester):
        """Can we print the tree?"""
        muts, mtree, mut_lvls = mtree_tester.get_muts_mtree()

        lvl_sets = [mut_lvls[:i] for i in range(1, len(mut_lvls))]
        for lvl_set in lvl_sets:
            for vals, _ in muts.groupby(lvl_set):
                print(mtree[vals])

        print(mtree)

    def test_iteration(self, mtree_tester):
        """Does the tree correctly implement iteration over subtrees?"""
        muts, mtree, mut_lvls = mtree_tester.get_muts_mtree()

        for nm, mut in mtree:
            assert nm in set(muts[mtree.mut_level])
            assert mut == mtree[nm]
            assert mut != mtree

        lvl_sets = {i: mut_lvls[:i] for i in range(1, len(mut_lvls))}
        for i, lvl_set in lvl_sets.items():

            for vals, _ in muts.groupby(lvl_set):
                if isinstance(vals, str):
                    vals = (vals,)

                for nm, mut in mtree[vals]:
                    assert nm in set(muts[mut_lvls[i]])
                    assert mut == mtree[vals][nm]
                    assert mut != mtree[vals[:-1]]

    def test_samples(self, mtree_tester):
        """Does the tree properly store its samples?"""
        muts, mtree, mut_lvls = mtree_tester.get_muts_mtree()

        for vals, mut in muts.groupby(mut_lvls):
            assert set(mtree[vals]) == set(mut['Sample'])

    def test_get_samples(self, mtree_tester):
        """Can we successfully retrieve the samples from the tree?"""
        muts, mtree, mut_lvls = mtree_tester.get_muts_mtree()

        lvl_sets = [mut_lvls[:i] for i in range(1, len(mut_lvls))]
        for lvl_set in lvl_sets:

            for vals, mut in muts.groupby(lvl_set):
                assert set(mtree[vals].get_samples()) == set(mut['Sample'])

    def test_allkeys(self, mtree_tester):
        """Can we retrieve the mutation set key of the tree?"""
        muts, mtree, mut_lvls = mtree_tester.get_muts_mtree()

        lvl_sets = chain.from_iterable(
            combn(mut_lvls, r)
            for r in range(1, len(mut_lvls) + 1))

        for lvl_set in lvl_sets:
            lvl_key = {}

            for vals, _ in muts.groupby(lvl_set):
                cur_key = lvl_key

                if isinstance(vals, str):
                    vals = (vals,)

                for i in range(len(lvl_set) - 1):
                    if (lvl_set[i], vals[i]) not in cur_key:
                        cur_key.update({(lvl_set[i], vals[i]): {}})
                    cur_key = cur_key[(lvl_set[i], vals[i])]

                cur_key.update({(lvl_set[-1], vals[-1]): None})

            assert mtree.allkey(lvl_set) == lvl_key


@pytest.mark.parametrize('mtype_tester', ['small', 'TP53', 'binary'],
                         indirect=True, scope="class")
class TestCaseBasicMuType:
    """Tests for basic functionality of MuTypes."""

    def test_print(self, mtype_tester):
        """Can we print MuTypes?"""
        mtypes = mtype_tester.get_types()

        for mtype in mtypes:
            print(mtype)

    def test_eq(self, mtype_tester):
        """Can we evaluate the equality of two MuTypes?"""
        mtypes = mtype_tester.get_types()

        for mtype in mtypes:
            assert mtype.__eq__(mtype)
        for mtype1, mtype2 in combn(mtypes, 2):
            assert mtype1 != mtype2

    def test_hash(self, mtype_tester):
        """Can we get proper hash values of MuTypes?"""
        mtypes = mtype_tester.get_types()

        for mtype1, mtype2 in product(mtypes, repeat=2):
            assert (mtype1 == mtype2) == (hash(mtype1) == hash(mtype2))

    def test_rawkey(self, mtype_tester):
        """Can we get the raw key of MuTypes?"""
        mtypes = mtype_tester.get_types()

        for mtype in mtypes:
            mtype.raw_key()


class TestCaseMuTypeBinary:
    """Tests the binary operators defined for MuTypes."""

    @pytest.mark.parametrize('mtype_tester', ['_all'],
                             indirect=True, scope="function")
    def test_invariants(self, mtype_tester):
        """Do binary operators preserve set theoretic invariants?"""
        mtypes = mtype_tester.get_types()

        for mtype in mtypes:
            assert mtype == (mtype & mtype)
            assert mtype == (mtype | mtype)
            assert (mtype - mtype) is None

        for mtype1, mtype2 in combn(mtypes, 2):
            if mtype1.get_levels() == mtype2.get_levels():
                assert not (mtype1 | mtype2) < (mtype1 & mtype2)

            if mtype1.get_levels() <= mtype2.get_levels():
                if mtype1 >= mtype2:
                    assert mtype2 == (mtype1 & mtype2)

            if mtype1.get_levels() >= mtype2.get_levels():
                if mtype1 <= mtype2:
                    assert mtype2 == (mtype1 | mtype2)

    @pytest.mark.parametrize('mtype_tester', ['small'],
                             indirect=True, scope="function")
    def test_or_easy(self, mtype_tester):
        """Can we take the union of two simple MuTypes?"""
        mtypes = mtype_tester.get_types()

        assert (mtypes[0] | mtypes[1]) == mtypes[0]
        assert (mtypes[0] | mtypes[4]) == mtypes[4]

        or_mtype = MuType({
            ('Gene', 'CDH1'): None,
            ('Gene', 'TTN'): {
                ('Form', 'Missense_Mutation'): {
                    ('Exon', ('326/363', '302/363', '10/363')): None
                    }}})
        assert (mtypes[1] | mtypes[2]) == or_mtype

    @pytest.mark.parametrize('mtype_tester', ['binary'],
                             indirect=True, scope="function")
    def test_or_hard(self, mtype_tester):
        """Can we take the union of two tricky MuTypes?"""
        mtypes = mtype_tester.get_types()

        assert (mtypes[0] | mtypes[1]) == mtypes[2]
        assert (mtypes[0] & mtypes[1]) == mtypes[3]

    @pytest.mark.parametrize('mtype_tester', ['small'],
                             indirect=True, scope="function")
    def test_and(self, mtype_tester):
        """Can we take the intersection of two MuTypes?"""
        mtypes = mtype_tester.get_types()

        assert (mtypes[0] & mtypes[1]) == mtypes[1]
        assert (mtypes[0] & mtypes[4]) == mtypes[0]

        and_mtype = MuType({
            ('Gene', 'TTN'): {
                ('Form', 'Missense_Mutation'): {
                    ('Exon', '10/363'): None
                    }}})
        assert (mtypes[1] & mtypes[2]) == and_mtype

    @pytest.mark.parametrize('mtype_tester', ['binary'],
                             indirect=True, scope="function")
    def test_sub(self, mtype_tester):
        """Can we subtract one MuType from another?"""
        mtypes = mtype_tester.get_types()

        sub_mtype = MuType({
            ('Gene', 'TP53'): {('Form', 'Missense_Mutation'): None}})
        assert (mtypes[2] - mtypes[0]) == sub_mtype


class TestCaseMuTypeSamples:
    """Tests for using MuTypes to access samples in MuTrees."""

    @pytest.mark.parametrize(
        ('mtree_tester', 'mtype_tester'),
        [(('small', ('Gene', 'Form', 'Exon')), 'small'),
         (('large', ('Gene', 'Form', 'Exon')), 'small'),
         (('large', ('Gene', 'Type', 'Form', 'Exon', 'Protein')), 'small'),
         (('TP53', ('Gene', 'Form', 'Exon')), 'small')],
        ids=muts_id, indirect=True, scope="function"
        )
    def test_basic(self, mtree_tester, mtype_tester):
        """Can we use basic MuTypes to get samples in MuTrees?"""
        muts, mtree, mut_lvls = mtree_tester.get_muts_mtree()
        mtypes = mtype_tester.get_types()

        assert (mtypes[0].get_samples(mtree)
                == set(muts['Sample'][muts['Gene'] == 'TTN']))
        assert (mtypes[3].get_samples(mtree)
                == set(muts['Sample'][muts['Form'] == 'Silent']))

        assert (mtypes[1].get_samples(mtree)
                == set(muts['Sample'][
                           (muts['Gene'] == 'TTN')
                           & (muts['Form'] == 'Missense_Mutation')
                           & ((muts['Exon'] == '326/363')
                              | (muts['Exon'] == '10/363'))
                           ]))

    @pytest.mark.parametrize(
        ('mtree_tester', 'mtype_tester'),
        [(('small', ('Gene', 'Form', 'Exon')), 'blank'),
         (('large', ('Gene', 'Form', 'Exon')), 'blank'),
         (('large', ('Gene', 'Protein', 'Exon')), 'blank')],
        ids=muts_id, indirect=True, scope="function"
        )
    def test_blank(self, mtree_tester, mtype_tester):
        """Are cases where no samples present properly handled?"""
        muts, mtree, mut_lvls = mtree_tester.get_muts_mtree()
        mtypes = mtype_tester.get_types()

        assert (mtypes[0].get_samples(mtree)
                == set(muts['Sample'][muts['Gene'] == 'TTN']))
        for mtype in mtypes[1:]:
            assert len(mtype.get_samples(mtree)) == 0

    @pytest.mark.parametrize(
        ('mtree_tester', 'mtype_tester'),
        [(('TP53', ('Gene', 'Form', 'Exon', 'Protein')), 'TP53')],
        ids=muts_id, indirect=True, scope="function"
        )
    def test_tp53(self, mtree_tester, mtype_tester):
        """Can subsets of TP53 be correctly retrieved?"""
        muts, mtree, mut_lvls = mtree_tester.get_muts_mtree()
        mtypes = mtype_tester.get_types()

        assert (mtypes[0].get_samples(mtree)
                == set(muts['Sample'][muts['Form'] == 'Splice_Site']))
        assert (mtypes[1].get_samples(mtree)
                == set(muts['Sample'][(muts['Exon'] == '7/11')
                                      | (muts['Exon'] == '8/11')]))
        assert (mtypes[2].get_samples(mtree)
                == set(muts['Sample'][(muts['Protein'] == 'p.R158L')
                                      | (muts['Protein'] == '.')]))

    @pytest.mark.parametrize(
        ('mtree_tester', 'mtype_tester'),
        [(('TP53', ('Gene', 'Form', 'Exon', 'Protein')), 'TP53')],
        ids=muts_id, indirect=True, scope="function"
        )
    def test_status(self, mtree_tester, mtype_tester):
        """Can we get a vector of mutation status from a MuTree?"""
        muts, mtree, mut_lvls = mtree_tester.get_muts_mtree()
        mtypes = mtype_tester.get_types()

        for mtype in mtypes:
            assert (mtree.status(['herpderp', 'derpherp'], mtype)
                    == [False, False])

        assert (
            mtree.status(['TCGA-04-1357-01A', 'dummy1', 'dummy2'],
                         MuType({('Protein', ('p.E224Gfs*4', 'nah')): None}))
            == [True, False, False]
            )

        samp_list = pd.Series(np.unique(muts['Sample']))
        assert (mtree.status(samp_list, mtypes[0])
                == samp_list.isin(
            muts['Sample'][muts['Form'] == 'Splice_Site'])).all()
        assert (mtree.status(samp_list, mtypes[1])
                == samp_list.isin(
            muts['Sample'][(muts['Exon'] == '7/11')
                           | (muts['Exon'] == '8/11')])).all()
        assert (mtree.status(samp_list, mtypes[2])
                == samp_list.isin(
            muts['Sample'][(muts['Protein'] == 'p.R158L')
                           | (muts['Protein'] == '.')])).all()


class TestCaseMuTreeLevels:
    """Tests for custom mutation levels."""

    @pytest.mark.parametrize('mtree_tester',
                             [('large', ('Gene', 'Type', 'Form', 'Protein'))],
                             ids=muts_id, indirect=True, scope="function")
    def test_type(self, mtree_tester):
        """Is the Type mutation level correctly defined?"""
        muts, mtree, mut_lvls = mtree_tester.get_muts_mtree()

        # Type level should catch all samples
        for (gene, form), mut in muts.groupby(['Gene', 'Form']):
            mtype = MuType({('Gene', gene): {('Form', form): None}})
            assert mtype.get_samples(mtree) == set(mut['Sample'])

        # Type level categories should be mutually exclusive
        test_key = mtree.allkey(levels=('Type', 'Protein'))
        assert (set(val for _, val in test_key.keys())
                <= {'CNV', 'Point', 'Frame', 'Other'})

        for plist1, plist2 in combn(test_key.values(), 2):
            assert not (set(val for _, val in plist1.keys())
                        & set(val for _, val in plist2.keys()))

    @pytest.mark.parametrize('mtree_tester',
                             [('large', ('Gene', 'Form_base', 'Form'))],
                             ids=muts_id, indirect=True, scope="function")
    def test_base_parse(self, mtree_tester):
        """Is the _base mutation level parser correctly defined?"""
        muts, mtree, mut_lvls = mtree_tester.get_muts_mtree()

        # _base level should catch all samples
        for (gene, form), mut in muts.groupby(['Gene', 'Form']):
            mtype = MuType({('Gene', gene): {('Form', form): None}})
            assert mtype.get_samples(mtree) == set(mut['Sample'])


class TestCaseAdvancedMuTree:
    """Tests for advanced functionality of MuTypes."""

    @pytest.mark.parametrize('mtype_tester', ['synonym'],
                             indirect=True, scope="function")
    def test_synonyms(self, mtype_tester):
        """Are MuTypes with synonymous mutation keys identical?"""
        mtypes = mtype_tester.get_types()

        assert mtypes[0] == mtypes[1]
        assert mtypes[2] == mtypes[3]
