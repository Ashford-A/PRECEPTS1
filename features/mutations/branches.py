
from functools import reduce
from operator import and_
from re import sub as gsub

from itertools import product
from itertools import permutations as perm


class MuType(object):
    """A set of properties recursively defining a particular type of mutation.

    This class corresponds to a mutation type defined through a list of
    properties, each possibly linked to a further mutation sub-type. Used in
    conjunction with the above MuTree class to represent and navigate the
    space of possible mutation subsets in a given cohort. While a MuTree is
    linked to a particular set of samples, a MuType represents a mutation
    type abstract of any samples that may or may not have it.

    MuTypes are initialized via recursively structured type dictionaries of
    the form type_dict={(Level, Label1): <None or subtype_dict>,
                        (Level, Label2): <None or subtype_dict>, ...}

    The keys of this type dictionary are thus 2-tuples composed of
        1) `Level`: anything that defines categories that a mutation can
        belong to, such as Gene, Exon, PolyPhen
        2) `Label`: one or more of these categories, such as TP53, 7/11, 0.67

    The value for a given key in a type_dict can be either `None`, indicating
    that all mutations in this category are represented this MuType, or
    another type_dict, indicating that only the given subset of mutations
    within this category are represented by this MuType. A MuType can thus
    contain children MuTypes, and the set of mutations the MuType stands for
    is the intersection of the children and their parent(s), and the union of
    parents and their siblings.

    For the sake of convenience, a `Label` can itself be a tuple of mutation
    property level categories, thus indicating that the corresponding subtype
    value applies to all the listed categories. Type dictionaries are
    automatically rationalized to group together identical subtype values when
    they are being parsed in order to reduce the memory footprint of each
    MuType object.

    Note that subtypes can be already-instantiated MuType objects instead of
    type dictionaries. Explicitly passing the `None` object by itself as a
    type dictionary creates a MuType corresponding to the empty null set of
    mutations, as does passing any empty iterable such as [] or (,).

    Arguments:
        type_dict (dict, list, tuple, None, or MuType)

    Attributes:
        cur_level (str): The mutation property level whose categories are
                         listed in this type.

    Examples:
        >>> # mutations of the KRAS gene
        >>> mtype1 = MuType({('Gene', 'KRAS'): None})
        >>>
        >>> # missense mutations of the KRAS gene
        >>> mtype2 = MuType({('Gene', 'KRAS'):
        >>>             {('Form', 'Missense_Mutation'): None}})
        >>>
        >>> # mutations of the BRAF or RB1 genes
        >>> mtype3 = MuType({('Gene', ('BRAF', 'RB1')): None})
        >>>
        >>> # frameshift mutations of the BRAF or RB1 genes and nonsense
        >>> # mutations of the TP53 gene occuring on its 8th exon
        >>> mtype4 = MuType({('Gene', ('BRAF', 'RB1')):
        >>>                     {('Type', 'Frame_Shift'): None},
        >>>                 {('Gene', 'TP53'):
        >>>                     {('Form', 'Nonsense_Mutation'):
        >>>                         {('Exon', '8/33'): None}}})

    """

    def __init__(self, type_dict):

        # ensures the type dictionary is in the proper format, parses out the
        # mutation property level from its keys
        if not type_dict:
            type_dict = {}
            self.cur_level = None

        elif isinstance(type_dict, dict):
            levels = set(lvl for lvl, _ in type_dict)

            if len(levels) > 1:
                raise ValueError("Improperly defined set key with multiple"
                                 "mutation levels!")

            else:
                self.cur_level = tuple(levels)[0]

        else:
            raise TypeError("MuType type dictionary must be a dict object!")

        # parses out the category labels listed for the given property level
        level_lbls = [(lbls, ) if isinstance(lbls, str) else lbls
                      for _, lbls in type_dict]

        # creates an expanded type dictionary where category labels that were
        # originally grouped together by subtype are given separate keys
        full_pairs = [
            (lbl, sub_type)
            for lbls, sub_type in zip(level_lbls, type_dict.values())
            if not (isinstance(sub_type, MuType) and sub_type.is_empty())
            for lbl in lbls
            ]

        # merges identical labels according to the union of their subtypes
        # i.e. silent:Exon7, silent:Exon8 => silent:(Exon7 or Exon8)
        full_dict = {}
        for lbl, sub_type in full_pairs:

            if lbl in full_dict:
                if sub_type is None or full_dict[lbl] is None:
                    full_dict[lbl] = None

                elif isinstance(sub_type, dict):
                    full_dict[lbl] |= MuType(sub_type)

                else:
                    full_dict[lbl] |= sub_type

            elif isinstance(sub_type, dict):
                full_dict[lbl] = MuType(sub_type)

            else:
                full_dict[lbl] = sub_type

        # collapses category labels with the same subtype into one key:subtype
        # pair, i.e. silent:None, frameshift:None => (silent, frameshift):None
        uniq_vals = [
            (frozenset(k for k, v in full_dict.items() if v == sub_type),
             sub_type)
            for sub_type in set(full_dict.values())
            ]

        # merges the subtypes of type dictionary entries with the same
        # category label, i.e. silent: <Exon IS 7/11>, silent: <Exon IS 10/11>
        # => silent: <Exon IS 7/11 or 10/11> to create the final dictionary
        self._child = {}
        for lbls, sub_type in uniq_vals:

            if lbls in self._child:
                if sub_type is None or self._child[lbls] is None:
                    self._child[lbls] = None

                else:
                    self._child[lbls] |= sub_type

            else:
                self._child[lbls] = sub_type

    def is_empty(self):
        """Checks if this MuType corresponds to the null mutation set."""
        return self._child == {}

    def get_levels(self):
        """Gets the property levels present in this type and its subtypes."""
        levels = {self.cur_level}

        for tp in self._child.values():
            if isinstance(tp, MuType):
                levels |= set(tp.get_levels())

        return levels

    def __hash__(self):
        """MuType hashes are defined in an analagous fashion to those of
           tuples, see for instance http://effbot.org/zone/python-hash.htm"""
        value = 0x163125 ^ len(self._child)

        for lbls, tp in sorted(self._child.items(), key=lambda x: list(x[0])):
            value += eval(hex((int(value) * 1000007) & 0xFFFFFFFF)[:-1])
            value ^= hash(lbls) ^ hash(tp)

        if value == -1:
            value = -2

        return value

    def child_iter(self):
        """Returns an iterator over the collapsed (labels):subtype pairs."""
        return self._child.items()

    def subtype_list(self):
        """Returns the list of all unique label:subtype pairs."""
        return [(lbl, tp) for lbls, tp in self._child.items() for lbl in lbls]

    def __len__(self):
        """Returns the number of unique category labels in the MuType.

        Examples:
            >>> len(MuType({('Gene', 'KRAS'): None}))
                1
            >>> len(MuType({('Gene', ('BRAF', 'TP53')): None}))
                2
            >>> len(MuType({('Gene', 'TTN'): {('Form', 'Nonsense'): None}}))
                1
            >>> len(MuType({('Gene', 'AR'): {('Exon', '2/87'): None},
            >>>             ('Gene', 'MUC16'): {('Exon', '4/8'): None}}))
                2

        """
        return len(self.subtype_list())

    def __eq__(self, other):
        """Checks if one MuType is equal to another."""

        # if the other object is not a MuType they are not equal
        if not isinstance(other, MuType):
            eq = False

        # MuTypes with different mutation property levels are not equal
        elif self.cur_level != other.cur_level:
            eq = False

        # MuTypes with the same mutation levels are equal if and only if
        # they have the same subtypes for the same level category labels
        else:
            eq = self.child_iter() == other.child_iter()

        return eq

    def __lt__(self, other):
        """Defines a sort order for MuTypes."""
        if not isinstance(other, MuType):
            return NotImplemented

        # we first compare the mutation property levels of the two MuTypes...
        if self.cur_level == other.cur_level:

            # sort label:subtype pairs according to label such that pairwise
            # invariance is preserved
            self_pairs = sorted(self.subtype_list(), key=lambda x: x[0])
            other_pairs = sorted(other.subtype_list(), key=lambda x: x[0])

            # ...then compare how many (label:subtype) pairs they have...
            if len(self_pairs) == len(other_pairs):
                self_lbls = [lbl for lbl, _ in self_pairs]
                other_lbls = [lbl for lbl, _ in other_pairs]

                # ...then compare the labels themselves...
                if self_lbls == other_lbls:
                    self_lvls = self.get_levels()
                    other_lvls = other.get_levels()

                    # ...then compare how deep the subtypes recurse...
                    if len(self_lvls) == len(other_lvls):
                        self_subtypes = [tp for _, tp in self_pairs]
                        other_subtypes = [tp for _, tp in other_pairs]

                        # ...then compare the subtypes for each pair of
                        # matching labels...
                        for tp1, tp2 in zip(self_subtypes, other_subtypes):
                            if tp1 != tp2:

                                # for the first pair of subtypes that are not
                                # equal (always the same pair because entries
                                # are sorted), we recursively compare the pair
                                if tp1 is None:
                                    return False

                                elif tp2 is None:
                                    return True

                                else:
                                    return tp1 < tp2

                        # if all subtypes are equal, the two MuTypes are equal
                        else:
                            return False

                    # MuTypes with fewer subtype levels are sorted first
                    else:
                        return len(self_lvls) < len(other_lvls)

                # MuTypes with different labels are sorted according to the
                # order defined by the sorted label lists
                else:
                    return self_lbls < other_lbls

            # MuTypes with fewer mutation entries are sorted first
            else:
                return len(self_pairs) < len(other_pairs)

        # MuTypes with differing mutation property levels are sorted according
        # to the sort order of the property strings
        else:
            return self.cur_level < other.cur_level

    # remaining methods necessary to define rich comparison for MuTypes
    def __ne__(self, other):
        return not self == other

    def __le__(self, other):
        return self < other or self == other

    def __gt__(self, other):
        return not (self < other or self == other)

    def __ge__(self, other):
        return not (self < other)

    def __repr__(self):
        """Shows the hierarchy of mutation properties within the MuType."""
        new_str = ''

        # iterate over all mutation types at this level separately
        # regardless of their children
        for lbl, tp in self.subtype_list():
            new_str += self.cur_level + ' IS ' + lbl

            if tp is not None:
                new_str += ' WITH ' + repr(tp)

            new_str += ' OR '

        return gsub(' OR $', '', new_str)

    def __str__(self):
        """Gets a condensed label for the MuType."""
        new_str = ''
        self_iter = sorted(
            self._child.items(),
            key=lambda x: sorted(list(x[0]))
            )

        # if there aren't too many types to list at this mutation level...
        if len(self_iter) <= 10:

            # ...iterate over the types, grouping together those with the
            # same children to produce a more concise label
            for lbls, tp in self_iter:

                if len(lbls) > 1:

                    #TODO: find a more elegant way of dealing with this
                    if self.cur_level == 'Gene' and tp is None:
                        new_str += '|'.join(sorted(lbls))

                    else:
                        new_str += '(' + '|'.join(sorted(lbls)) + ')'

                else:
                    new_str += list(lbls)[0]

                if tp is not None:
                    new_str += ':' + str(tp)

                new_str += '|'

        # ...otherwise, show how many types there are and move on to the
        # levels further down if they exist
        else:
            new_str += "({} {}s)".format(
                len(self.subtype_list()), self.cur_level.lower())

            # condense sub-types at the further levels
            for lbls, tp in self_iter:
                new_str += "-(>= {} sub-types at level(s): {})".format(
                    len(tp),
                    reduce(lambda x, y: x + ', ' + y,
                           self.get_levels() - {self.cur_level})
                    )

        return gsub('\\|+$', '', new_str)

    def get_label(self):
        return gsub('/|\.|:', '_', str(self))

    def __or__(self, other):
        """Returns the union of two MuTypes."""
        if not isinstance(other, MuType):
            return NotImplemented

        if self.is_empty():
            return other
        if other.is_empty():
            return self

        # gets the unique label:subtype pairs in each MuType
        self_dict = dict(self.subtype_list())
        other_dict = dict(other.subtype_list())

        if self.cur_level == other.cur_level:
            new_key = {}

            # adds the subtypes paired with the labels in the symmetric
            # difference of the labels in the two MuTypes
            new_key.update({(self.cur_level, lbl): self_dict[lbl]
                            for lbl in self_dict.keys() - other_dict.keys()})
            new_key.update({(self.cur_level, lbl): other_dict[lbl]
                            for lbl in other_dict.keys() - self_dict.keys()})

            # finds the union of the subtypes paired with each of the labels
            # appearing in both MuTypes
            new_key.update(
                {(self.cur_level, lbl): (
                    None if self_dict[lbl] is None or other_dict[lbl] is None
                    else self_dict[lbl] | other_dict[lbl]
                    )
                 for lbl in self_dict.keys() & other_dict.keys()}
                )

        else:
            raise ValueError(
                "Cannot take the union of two MuTypes with "
                "mismatching mutation levels {} and {}!".format(
                    self.cur_level, other.cur_level)
                )

        return MuType(new_key)

    def __and__(self, other):
        """Finds the intersection of two MuTypes."""
        if not isinstance(other, MuType):
            return NotImplemented

        if self.is_empty() or other.is_empty():
            return MuType({})

        # gets the unique label:subtype pairs in each MuType
        self_dict = dict(self.subtype_list())
        other_dict = dict(other.subtype_list())

        new_key = {}
        if self.cur_level == other.cur_level:
            for lbl in self_dict.keys() & other_dict.keys():

                if self_dict[lbl] is None:
                    new_key.update({(self.cur_level, lbl): other_dict[lbl]})

                elif other_dict[lbl] is None:
                    new_key.update({(self.cur_level, lbl): self_dict[lbl]})

                else:
                    new_ch = self_dict[lbl] & other_dict[lbl]

                    if not new_ch.is_empty():
                        new_key.update({(self.cur_level, lbl): new_ch})

        elif other.cur_level in self.get_levels():
            for lbl in self_dict.keys():

                if self_dict[lbl] is None:
                    new_key.update({(self.cur_level, lbl): other})

                else:
                    new_ch = self_dict[lbl] & other

                    if not new_ch.is_empty():
                        new_key.update({(self.cur_level, lbl): new_ch})

        else:
            for lbl in self_dict.keys():

                if self_dict[lbl] is None:
                    new_key.update({(self.cur_level, lbl): other})

                else:
                    new_ch = other & self_dict[lbl]

                    if not new_ch.is_empty():
                        new_key.update({(self.cur_level, lbl): new_ch})

        return MuType(new_key)

    def __add__(self, other):
        """The sum of MuTypes yields the type where both MuTypes appear."""
        if not isinstance(other, MuType):
            return NotImplemented

        return MutComb([self, other])

    def is_supertype(self, other):
        """Checks if one MuType (non-strictly) contains another MuType."""
        if not isinstance(other, MuType):
            return NotImplemented

        # the empty null set cannot be the supertype of any other MuType
        if self.is_empty():
            return False

        # the empty null set is a subtype of every other non-empty MuType
        if other.is_empty():
            return True

        # gets the unique label:subtype pairs in each MuType
        self_dict = dict(self.subtype_list())
        other_dict = dict(other.subtype_list())

        # a MuType cannot be a supertype of another MuType unless they are on
        # the same mutation property level and its category labels are a
        # superset of the others'
        if self.cur_level == other.cur_level:
            if self_dict.keys() >= other_dict.keys():

                for k in (self_dict.keys() & other_dict.keys()):
                    if self_dict[k] is not None:

                        if other_dict[k] is None:
                            return False
                        elif not self_dict[k].is_supertype(other_dict[k]):
                            return False

            else:
                return False

        else:
            return False

        return True

    def __sub__(self, other):
        """Subtracts one MuType from another."""
        if not isinstance(other, MuType):
            return NotImplemented

        if self.is_empty() or other.is_empty():
            return self

        # gets the unique label:subtype pairs in each MuType
        self_dict = dict(self.subtype_list())
        other_dict = dict(other.subtype_list())

        if self.cur_level == other.cur_level:
            new_key = {(self.cur_level, lbl): self_dict[lbl]
                       for lbl in self_dict.keys() - other_dict.keys()}

            for lbl in self_dict.keys() & other_dict.keys():
                if other_dict[lbl] is not None:

                    if self_dict[lbl] is not None:
                        sub_val = self_dict[lbl] - other_dict[lbl]

                        if not sub_val.is_empty():
                            new_key.update({(self.cur_level, lbl): sub_val})

                    else:
                        new_key.update({(self.cur_level, lbl): None})

        else:
            raise ValueError("Cannot subtract MuType with mutation level {} "
                             "from MuType with mutation level {}!".format(
                                other.cur_level, self.cur_level))

        return MuType(new_key)

    def get_samples(self, mtree):
        """Gets the samples contained in branch(es) of a MuTree.

        Args:
            mtree (HetMan.features.mutations.trees.MuTree): A hierarchy of mutations present in a cohort.

        Returns:
            samps (set): The samples in the MuTree that have the mutation(s)
                         specified by this MuType.
                         .
        """

        # if this MuType has the same mutation level as the MuTree...
        samps = set()
        if self.cur_level == mtree.mut_level:

            # ...find the mutation entries in the MuTree that match the
            # mutation entries in the MuType
            for (nm, mut), (lbl, tp) in product(mtree, self.subtype_list()):
                if lbl == nm:

                    if isinstance(mut, frozenset):
                            samps |= mut

                    else:
                        if tp is None:
                            samps |= mut.get_samples()
                        else:
                            samps |= tp.get_samples(mut)

        else:
            for _, mut in mtree:
                if (not isinstance(mut, frozenset)
                        and mut.get_levels() & self.get_levels()):
                    samps |= self.get_samples(mut)

        return samps

    def invert(self, mtree):
        """Gets the MuType of mutations in a MuTree but not in this MuType.

        Args:
            mtree (MuTree): A hierarchy of mutations present in a cohort.

        Returns:
            inv_mtype (MuType)

        """
        return mtree.get_diff(MuType(mtree.allkey()), self)

    def subkeys(self):
        """Gets all of the possible subsets of this MuType that contain
           exactly one of the leaf properties."""
        mkeys = []

        for lbls, tp in list(self.child_iter()):
            if tp is None:
                mkeys += [{(self.cur_level, lbl): None} for lbl in lbls]

            else:
                mkeys += [{(self.cur_level, lbl): sub_tp}
                          for lbl in lbls for sub_tp in tp.subkeys()]

        return mkeys


class MutComb(object):
    """A class corresponding to the presence of multiple mutation sub-types.

    Arguments:
        mtypes (:obj:`list` of :obj:`MuType`)

    Examples:
        >>> # create an object representing samples that have both a TTN
        >>> # mutation and a KRAS missense mutation
        >>>
        >>> from HetMan.features.mutations.branches import MuType
        >>>
        >>> mtype1 = MuType({('Gene', 'TTN'): None})
        >>> mtype2 = MuType({('Gene', 'KRAS'): {
        >>>                     ('Form', 'Missense_Mutation'): None}})
        >>>
        >>> comb_type = MutComb([mtype1, mtype2])

    """

    def __new__(cls, mtypes):
        if not all(isinstance(mtype, MuType) for mtype in mtypes):
            raise TypeError(
                "A MutComb object must be a combination of MuTypes!")

        obj = super().__new__(cls)
        mtypes = list(mtypes)

        for i, j in perm(range(len(mtypes)), r=2):
            if mtypes[i] and mtypes[j]:
                mtypes[j] -= mtypes[i]

        mtypes = [mtype for mtype in mtypes if mtype and not mtype.is_empty()]

        if mtypes:
            if len(mtypes) == 1:
                return mtypes[0]
            else:
                obj.mtypes = frozenset(mtypes)
                return obj

    def mtype_apply(self, each_fx, comb_fx):
        each_list = [each_fx(mtype) for mtype in self.mtypes]
        return reduce(comb_fx, each_list)

    def __repr__(self):
        return self.mtype_apply(repr, lambda x, y: x + ' AND ' + y)

    def __str__(self):
        return self.mtype_apply(str, lambda x, y: x + ' & ' + y)

    def __hash__(self):
        value = 0x213129

        return value + self.mtype_apply(
            hash,
            lambda x, y: (x ^ y + eval(hex((int(value) * 1003)
                                           & 0xFFFFFFFF)[:-1]))
            )

    def get_samples(self, mtree):
        return self.mtype_apply(lambda x: x.get_samples(mtree), and_)
