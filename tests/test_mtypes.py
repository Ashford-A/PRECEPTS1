
"""Unit tests for abstract representations of mutation sub-types.

See Also:
    :class:`..features.variants,MuType`: Contains the class tested herein.

Author: Michal Grzadkowski <grzadkow@ohsu.edu>

"""

from ..features.variants import MuType
from .resources import *
import pytest

from functools import reduce
from operator import add

from itertools import combinations as combn
from itertools import product


# .. objects that are re-used across multiple tests ..
@pytest.fixture(scope='function')
def mtype_tester(request):
    """Create a set of mutation subtypes."""
    return MuTypeTester(request.param)


class MuTypeTester(object):
    """Defines a set of mutation subtypes for use in testing.

    Attributes:
        type_lbl (str): Corresponds to a set of subtypes present in the
                        TypeTester.mtypes object. Can be 'ALL' in which case
                        all subtypes are loaded together.

    Examples:
        >>> MuTypeTester('small')
        >>> MuTypeTester('synonym')
        >>> MuTypeTester('binary_basic')
        >>> MuTypeTester('ALL')

    """

    def __init__(self, request):
        self.type_lbls = request.split('_')

    def get_types(self):
        """Loads the mutation subtype sets defined by this instance."""

        if "ALL" in self.type_lbls:
            return reduce(
                add,
                [tps for _, tps in vars(test_mtypes).items()
                 if isinstance(tps, tuple) and isinstance(tps[0], MuType)]
                )

        else:
            return reduce(add,
                          [eval('test_mtypes.{}'.format(lbl))
                           for lbl in self.type_lbls])


class TestCaseInit:
    """Tests for proper instatiation of MuTypes from type dictionaries."""

    @pytest.mark.parametrize('mtype_tester', ['basic'],
                             indirect=True, scope="function")
    def test_child(self, mtype_tester):
        """Is the child attribute of a MuType properly created?"""
        mtypes = mtype_tester.get_types()

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

    @pytest.mark.parametrize('mtype_tester', ['basic'],
                             indirect=True, scope="function")
    def test_level(self, mtype_tester):
        assert all(mtype.cur_level == 'Gene'
                   for mtype in mtype_tester.get_types())

    @pytest.mark.parametrize('mtype_tester', ['basic'],
                             indirect=True, scope="function")
    def test_len(self, mtype_tester):
        assert [len(mtype) for mtype in mtype_tester.get_types()].__eq__(
            [1, 1, 2, 2, 2, 2])


@pytest.mark.parametrize('mtype_tester', ['ALL'],
                         indirect=True, scope="class")
class TestCaseBasic:
    """Tests for basic functionality of MuTypes."""

    def test_iter(self, mtype_tester):
        """Can we iterate over the sub-types in a MuType?"""
        for mtype in mtype_tester.get_types():
            assert len(mtype.subtype_list()) >= len(list(mtype.child_iter()))

    def test_print(self, mtype_tester):
        """Can we print MuTypes?"""
        mtypes = mtype_tester.get_types()

        for mtype in mtypes:
            assert isinstance(repr(mtype), str)
            assert isinstance(str(mtype), str)

        for mtype1, mtype2 in combn(mtypes, 2):
            if mtype1 == mtype2:
                assert str(mtype1) == str(mtype2)
            else:
                assert repr(mtype1) != repr(mtype2)
                assert str(mtype1) != str(mtype2)

    def test_hash(self, mtype_tester):
        """Can we get proper hash values of MuTypes?"""
        mtypes = mtype_tester.get_types()

        for mtype1, mtype2 in product(mtypes, repeat=2):
            assert (mtype1 == mtype2) == (hash(mtype1) == hash(mtype2))


@pytest.mark.parametrize('mtype_tester', ['sorting'],
                         indirect=True, scope="class")
class TestCaseSorting:
    """Tests the sort order defined for MuTypes."""

    def test_sort(self, mtype_tester):
        """Can we correctly sort a list of MuTypes?"""
        mtypes = mtype_tester.get_types()

        assert sorted(mtypes) == [
            mtypes[5], mtypes[6], mtypes[7], mtypes[2], mtypes[1], mtypes[10],
            mtypes[3], mtypes[9], mtypes[11], mtypes[0], mtypes[4], mtypes[8]
            ]

    def test_sort_invariance(self, mtype_tester):
        """Is the sort order for MuTypes consistent?"""
        mtypes = mtype_tester.get_types()

        for mtype1, mtype2, mtype3 in combn(mtypes, 3):
            assert not (mtype1 < mtype2 < mtype3 < mtype1)

        assert sorted(mtypes) == sorted(list(reversed(mtypes)))
        assert sorted(mtypes[0:11:2]) == [
            mtypes[6], mtypes[2], mtypes[10], mtypes[0], mtypes[4], mtypes[8]
            ]
        assert sorted(mtypes[:6]) == [
            mtypes[5], mtypes[2], mtypes[1], mtypes[3], mtypes[0], mtypes[4],
            ]


class TestCaseBinary:
    """Tests the binary operators defined for MuTypes."""

    @pytest.mark.parametrize('mtype_tester', ['ALL'],
                             indirect=True, scope="function")
    def test_comparison(self, mtype_tester):
        """Are rich comparison operators correctly implemented for MuTypes?"""
        mtypes = mtype_tester.get_types()

        for mtype in mtypes:
            assert mtype == mtype

            assert mtype <= mtype
            assert mtype >= mtype
            assert not mtype < mtype
            assert not mtype > mtype


        for mtype1, mtype2 in combn(mtypes, 2):
            assert (mtype1 <= mtype2) or (mtype1 > mtype2)

            if mtype1 < mtype2:
                assert mtype1 <= mtype2
                assert mtype1 != mtype2

            elif mtype1 > mtype2:
                assert mtype1 >= mtype2
                assert mtype1 != mtype2

    @pytest.mark.parametrize('mtype_tester', ['ALL'],
                             indirect=True, scope="function")
    def test_invariants(self, mtype_tester):
        """Do binary operators preserve set theoretic invariants?"""
        mtypes = mtype_tester.get_types()

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

