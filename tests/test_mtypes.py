
"""Unit tests for abstract representations of mutation sub-types.

See Also:
    :class:`..features.mutations.MuType`: Contains the class tested herein.

Author: Michal Grzadkowski <grzadkow@ohsu.edu>

"""

from ..features.mutations import MuType
from .resources import *
import pytest

from functools import reduce
from operator import add, or_, and_

from itertools import combinations as combn
from itertools import product


def mtype_tester(mtypes_param):
    type_lbls = mtypes_param.split('_')

    if "ALL" in type_lbls:
        mtypes = reduce(
            add,
            [tps for _, tps in vars(test_mtypes).items()
             if isinstance(tps, tuple) and isinstance(tps[0], MuType)]
            )

    else:
        mtypes = reduce(add, [eval('test_mtypes.{}'.format(lbl))
                              for lbl in type_lbls])

    return mtypes


def pytest_generate_tests(metafunc):

    if metafunc.function.__code__.co_argcount > 1:
        if hasattr(metafunc.cls, 'params'):
            if isinstance(metafunc.cls.params, dict):
                funcarglist = metafunc.cls.params[metafunc.function.__name__]

            else:
                funcarglist = metafunc.cls.params

        else:
            funcarglist = 'ALL'

        if isinstance(funcarglist, str):
            funcarglist = [funcarglist]

        metafunc.parametrize('mtypes', [mtype_tester(funcarg)
                                        for funcarg in funcarglist])


class TestCaseInit(object):
    """Tests for proper instatiation of MuTypes from type dictionaries."""

    params = {'test_child': 'basic',
              'test_levels': 'basic',
              'test_len': 'basic'}

    def test_child(self, mtypes):
        """Is the child attribute of a MuType properly created?"""

        assert mtypes[0]._child == {frozenset(['TP53']): None}
        assert mtypes[1]._child == {
            frozenset(['TP53']): MuType({('Form', 'Frame_Shift'): None})}

        assert mtypes[2]._child == {frozenset(['TP53', 'TTN']): None}
        assert mtypes[3]._child == mtypes[2]._child
        assert mtypes[4]._child == mtypes[2]._child
        assert mtypes[5]._child == mtypes[2]._child

    def test_empty(self):
        """Can we correctly instantiate an empty MuType?"""

        assert MuType(None).is_empty()
        assert MuType({}).is_empty()
        assert MuType([]).is_empty()
        assert MuType(()).is_empty()

    def test_levels(self, mtypes):
        """Do mutations store mutation annotation levels correctly?"""

        assert all(mtype.cur_level == 'Gene'
                   for mtype in mtypes)

        assert mtypes[0].get_levels() == {'Gene'}
        assert mtypes[1].get_levels() == {'Gene', 'Form'}
        assert mtypes[2].get_levels() == {'Gene'}
        assert mtypes[3].get_levels() == {'Gene'}
        assert mtypes[4].get_levels() == {'Gene'}
        assert mtypes[5].get_levels() == {'Gene'}

    def test_len(self, mtypes):
        assert [len(mtype) for mtype in mtypes].__eq__([1, 1, 2, 2, 2, 2])


class TestCaseBasic:
    """Tests for basic functionality of MuTypes."""

    params = 'ALL'

    def test_iter(self, mtypes):
        """Can we iterate over the sub-types in a MuType?"""
        for mtype in mtypes:
            assert len(mtype.subtype_list()) >= len(list(mtype.child_iter()))

    def test_print(self, mtypes):
        """Can we print MuTypes?"""
        for mtype in mtypes:
            assert isinstance(repr(mtype), str)
            assert isinstance(str(mtype), str)

        for mtype1, mtype2 in combn(mtypes, 2):
            if mtype1 == mtype2:
                assert str(mtype1) == str(mtype2)

            else:
                assert repr(mtype1) != repr(mtype2)
                assert str(mtype1) != str(mtype2)

    def test_hash(self, mtypes):
        """Can we get proper hash values of MuTypes?"""
        for mtype1, mtype2 in product(mtypes, repeat=2):
            assert (mtype1 == mtype2) == (hash(mtype1) == hash(mtype2))

    def test_subkeys(self, mtypes):
        """Can we get the leaf types stored in a MuType?"""
        for mtype in mtypes:
            key_mtypes = [MuType(k) for k in mtype.subkeys()]

            assert len(set(key_mtypes)) == len(key_mtypes)
            assert reduce(or_, key_mtypes) == mtype

            if len(key_mtypes) > 1:
                assert reduce(and_, key_mtypes).is_empty()

        for mtype1, mtype2 in combn(mtypes, 2):
            if mtype1 == mtype2:
                keys1 = mtype1.subkeys()
                keys2 = mtype2.subkeys()

                assert len(keys1) == len(keys2)
                assert (sorted(MuType(k) for k in keys1)
                        == sorted(MuType(k) for k in keys2))


class TestCaseSorting:
    """Tests the sort order defined for MuTypes."""

    params = {'test_sort': 'sorting',
              'test_sort_invariance': 'ALL'}

    def test_sort(self, mtypes):
        """Can we correctly sort a list of MuTypes?"""
        assert sorted(mtypes) == [
            mtypes[5], mtypes[6], mtypes[7], mtypes[2], mtypes[1], mtypes[10],
            mtypes[3], mtypes[9], mtypes[11], mtypes[0], mtypes[4], mtypes[8]
            ]

        assert sorted(mtypes[0:11:2]) == [
            mtypes[6], mtypes[2], mtypes[10], mtypes[0], mtypes[4], mtypes[8]
            ]

        assert sorted(mtypes[:6]) == [
            mtypes[5], mtypes[2], mtypes[1], mtypes[3], mtypes[0], mtypes[4],
            ]

    def test_sort_invariance(self, mtypes):
        """Is the sort order for MuTypes consistent?"""
        for mtype1, mtype2, mtype3 in combn(mtypes, 3):
            assert not (mtype1 < mtype2 < mtype3 < mtype1)

        assert sorted(mtypes) == sorted(list(reversed(mtypes)))


class TestCaseBinary:
    """Tests the binary operators defined for MuTypes."""

    params = {'test_comparison': 'ALL',
              'test_invariants': 'ALL',
              'test_or_easy': 'small',
              'test_or_hard': 'binary',
              'test_and': 'small',
              'test_sub': 'binary'}

    def test_comparison(self, mtypes):
        """Are rich comparison operators correctly implemented for MuTypes?"""
        for mtype in mtypes:
            assert mtype == mtype

            assert mtype <= mtype
            assert mtype >= mtype
            assert not mtype < mtype
            assert not mtype > mtype


        for mtype1, mtype2 in combn(mtypes, 2):
            assert (mtype1 <= mtype2) != (mtype1 > mtype2)

            if mtype1 < mtype2:
                assert mtype1 <= mtype2
                assert mtype1 != mtype2

            elif mtype1 > mtype2:
                assert mtype1 >= mtype2
                assert mtype1 != mtype2

    def test_invariants(self, mtypes):
        """Do binary operators preserve set theoretic invariants?"""
        for mtype in mtypes:
            assert mtype == (mtype & mtype)
            assert mtype == (mtype | mtype)
            assert (mtype - mtype).is_empty()

        for mtype1, mtype2 in combn(mtypes, 2):

            if mtype1.get_levels() == mtype2.get_levels():
                assert mtype1 | mtype2 == mtype2 | mtype1
                assert mtype1 & mtype2 == mtype2 & mtype1

                assert (mtype1 | mtype2).is_supertype(mtype1 & mtype2)
                assert mtype1 - mtype2 == mtype1 - (mtype1 & mtype2)
                assert mtype1 | mtype2 == (
                    (mtype1 - mtype2) | (mtype2 - mtype1)
                    | (mtype1 & mtype2)
                    )

            if mtype1.get_levels() <= mtype2.get_levels():
                if mtype1 == mtype2 or mtype1.is_supertype(mtype2):
                    assert mtype2 == (mtype1 & mtype2)

            if mtype1.get_levels() >= mtype2.get_levels():
                if mtype1 == mtype2 or mtype2.is_supertype(mtype1):
                    assert mtype2 == (mtype1 | mtype2)

    def test_or_easy(self, mtypes):
        """Can we take the union of two simple MuTypes?"""
        assert (mtypes[0] | mtypes[1]) == mtypes[0]
        assert (mtypes[0] | mtypes[4]) == mtypes[4]

        or_mtype = MuType({
            ('Gene', 'CDH1'): None,
            ('Gene', 'TTN'): {
                ('Form', 'Missense_Mutation'): {
                    ('Exon', ('326/363', '302/363', '10/363')): None
                    }}})
        assert (mtypes[1] | mtypes[2]) == or_mtype

    def test_or_hard(self, mtypes):
        """Can we take the union of two tricky MuTypes?"""
        assert (mtypes[0] | mtypes[1]) == mtypes[2]
        assert (mtypes[0] & mtypes[1]) == mtypes[3]

    def test_and(self, mtypes):
        """Can we take the intersection of two MuTypes?"""
        assert (mtypes[0] & mtypes[1]) == mtypes[1]
        assert (mtypes[0] & mtypes[4]) == mtypes[0]

        and_mtype = MuType({
            ('Gene', 'TTN'): {
                ('Form', 'Missense_Mutation'): {
                    ('Exon', '10/363'): None
                    }}})
        assert (mtypes[1] & mtypes[2]) == and_mtype

    def test_sub(self, mtypes):
        """Can we subtract one MuType from another?"""
        sub_mtype = MuType({
            ('Gene', 'TP53'): {('Form', 'Missense_Mutation'): None}})
        assert (mtypes[2] - mtypes[0]) == sub_mtype
