
"""Loading and processing variant datasets.

This file contains functions and classes for loading, processing, and storing
mutations such as SNPs, indels, and frameshifts in formats suitable for use
in machine learning pipelines.

See Also:
    :module:`.copies`: Dealing with copy number alterations.

Author: Michal Grzadkowski <grzadkow@ohsu.edu>

"""

import numpy as np
import pandas as pd

import json
from re import sub as gsub
from math import exp
from ophion import Ophion

from functools import reduce
from itertools import combinations as combn

from sklearn.cluster import MeanShift


# .. functions for loading mutation data from external data sources ..
def get_variants_mc3(syn):
    """Reads ICGC mutation data from the MC3 synapse file.

    Args:
        syn (Synapse): A logged-in synapseclient instance.

    Returns:
        muts (pandas DataFrame), shape = (n_mutations, mut_levels+1)
            An array of mutation data, with a row for each mutation
            appearing in an individual sample.

    Examples:
        >>> import synapseclient
        >>> syn = synapseclient.Synapse()
        >>> syn.login()
        >>> muts = get_variants_mc3(syn)

    """
    mc3 = syn.get('syn7824274')

    # defines which mutation annotation MAF columns to use
    use_cols = [0, 8, 15, 36, 38, 72]
    use_names = ['Gene', 'Form', 'Sample', 'Protein', 'Exon', 'PolyPhen']

    # imports mutation data into a DataFrame, parses TCGA sample barcodes
    # and PolyPhen scores
    muts = pd.read_csv(mc3.path, usecols=use_cols, sep='\t', header=None,
                       names=use_names, comment='#', skiprows=1)
    muts['Sample'] = [reduce(lambda x, y: x + '-' + y, s.split('-', 4)[:4])
                      for s in muts['Sample']]
    muts['PolyPhen'] = [gsub('\)$', '', gsub('^.*\(', '', x))
                        if x != '.' else 0 for x in muts['PolyPhen']]

    return muts


def get_variants_bmeg(sample_list, gene_list, mut_fields=("term", )):
    """Gets variants from BMEG."""

    oph = Ophion("http://bmeg.io")
    mut_list = {samp: {} for samp in sample_list}
    gene_lbls = ["gene:" + gn for gn in gene_list]

    print(oph.query().has("gid", "biosample:" + sample_list[0])
          .incoming("variantInBiosample")
          .outEdge("variantInGene").mark("variant")
          .inVertex().has("gid", oph.within(gene_lbls)).count().execute())
          # .mark("gene").select(["gene", "variant"]).count().execute())

    for samp in sample_list:
        for i in oph.query().has("gid", "biosample:" + samp)\
                .incoming("variantInBiosample")\
                .outEdge("variantInGene").mark("variant")\
                .inVertex().has("gid", oph.within(gene_lbls))\
                .mark("gene").select(["gene", "variant"]).execute():
            dt = json.loads(i)
            gene_name = dt["gene"]["properties"]["symbol"]
            mut_list[samp][gene_name] = {
                k: v for k, v in dt["variant"]["properties"].items()
                if k in mut_fields}

    mut_table = pd.DataFrame(mut_list)

    return mut_table


class MuTree(object):
    """A hierarchy of samples organized by mutation annotation levels.

    A MuTree stores variant mutant data for a set of samples in a tree-like
    data structure. Each level in the tree corresponds to a particular
    mutation annotation hierarchy level, such as Gene, Form, Exon, Protein,
    etc. Each node in the tree corresponds to a particular value of the
    annotation level present in at least one of the samples stored in the
    tree. A node at the ith level of the tree has children nodes for each of
    the values present at the (i+1)th annotation level for the samples having
    mutations with the ith level node value.

    Every node in a MuTree is also a MuTree, except for the leaf nodes, which
    are frozensets of the samples which have all of the annotation level
    values of their parent nodes. Thus a MuTree with mutation levels ['Gene',
    'Exon'] may have a 'KRAS' and 'TP53' nodes at the top level, which have
    children nodes ('4', '5') and ('7'), which list the samples having
    KRAS mutations on the 4th exon, KRAS mutations on the 5th exon, and TP53
    mutations on the 7th exon respectively.

    Levels can either be fields in the 'muts' DataFrame, in which case
    the tree will have a branch for each unique value in the field, or
    one of the keys of the MuTree.mut_fields object, in which case they
    will be defined by the corresponding MuType.muts_<level> method.

    Attributes:
        depth (int): How many mutation levels are above the tree
                     in the hierarchy.
        mut_level (str): The mutation annotation level described by the top
                         level of the tree.

    Args:
        muts (pandas DataFrame), shape = [n_muts, ]
            Input mutation data, each record is a mutation occurring in
            a sample to be included in the tree.
            Must contain a 'Sample' column.
        
        levels (tuple of str):
            A list of mutation annotation levels to be included in the tree.

    Examples:
        >>> mut_data = pd.DataFrame(
        >>>     {'Sample': ['S1', 'S2', 'S3', 'S4'],
        >>>      'Gene': ['TP53', 'TP53', 'KRAS', 'TP53'],
        >>>      'Exon': ['3', '3', '2', '7'],
        >>>      'Protein': ['H3R', 'S7T', 'E1R', 'Y11R']}
        >>>     )
        >>> mtree = MuTree(mut_data, levels=['Gene', 'Exon', 'Protein'])
        >>> print(mtree)
            Gene IS TP53 AND
                Exon is 3 AND
                    Protein is H3R: S1
                    Protein is S7T: S2
                Exon is 7 AND
                    Protein is Y11R: S4
            Gene is KRAS AND
                Exon is 2 AND
                    Protein is E1R: S3

    """

    # mapping between fields in an input mutation table and
    # custom mutation levels
    mut_fields = {
        'Type': ('Gene', 'Form', 'Protein'),
        'Location': ('Protein', ),
        }

    # .. functions for finding available branches of mutation levels ..
    @classmethod
    def check_muts(cls, muts, levels):
        """Checks that at least one of the given levels can be found in the
           given list of mutations.
        """
        muts_left = False
        lvls_left = list(levels)

        while lvls_left and not muts_left:
            cur_lvl = lvls_left.pop(0).split('_')[0]
            if cur_lvl in muts:
                muts_left = not np.all(pd.isnull(muts[cur_lvl]))

            elif cur_lvl in cls.mut_fields:
                if not np.all([x in muts for x in cls.mut_fields[cur_lvl]]):
                    raise ValueError("For mutation level " + cur_lvl + ", "
                                     + str(cls.mut_fields[cur_lvl])
                                     + " need to be provided as fields.")
                else:
                    muts_left = not np.all(pd.isnull(
                        muts.loc[:, cls.mut_fields[cur_lvl]]))

            else:
                raise ValueError("Unknown mutation level " + cur_lvl
                                 + " which is not in the given mutation data"
                                 + " frame and not a custom-defined level!")

        return muts_left

    @classmethod
    def split_muts(cls, muts, lvl_name):
        """Splits mutations into tree branches for a given level."""

        # level names have to consist of a base level name and an optional
        # parsing label separated by an underscore
        lvl_info = lvl_name.split('_')
        if len(lvl_info) > 2:
            raise ValueError("Invalid level name " + lvl_name
                             + " with more than two fields!")

        # if a parsing label is present, add the parsed level
        # to the table of mutations
        elif len(lvl_info) == 2:
            parse_lbl = lvl_info[1].lower()
            parse_fx = 'parse_' + parse_lbl

            if parse_fx in cls.__dict__:
                muts = eval('cls.' + parse_fx)(muts, lvl_info[0])

            else:
                raise ValueError("Custom parse label " + parse_lbl + " must "
                                 + "have a corresponding <" + parse_fx +
                                 "> method defined in " + cls.__name__ + "!")

        # splits mutations according to values of the specified level
        if isinstance(muts, tuple):
            if np.all(pd.isnull(val) for _, val in muts):
                split_muts = {}
            else:
                split_muts = muts
        elif lvl_info[0] in muts:
            split_muts = dict(tuple(muts.groupby(lvl_info[0])))

        # if the specified level is not a column in the mutation table,
        # we assume it's a custom mutation level
        else:
            split_fx = 'muts_' + lvl_info[0].lower()
            if split_fx in cls.__dict__:
                split_muts = eval('cls.' + split_fx)(muts)
            else:
                raise ValueError("Custom mutation level " + lvl_name
                                     + " must have a corresponding <"
                                     + split_fx + "> method defined in "
                                     + cls.__name__ + "!")

        return split_muts

    # .. functions for defining custom mutation levels ..
    @staticmethod
    def muts_type(muts):
        """Parses mutations according to Type, which can be 'CNV' (Gain or
           Loss), 'Point' (missense and silent mutations), or 'Frame' (indels,
           frameshifts, nonsense mutations).
        """
        new_muts = {}

        cnv_indx = muts['Form'].isin(['Gain', 'Loss'])
        point_indx = muts['Protein'].str.match(
            pat='^p\\.[A-Z][0-9]+[A-Z]$', as_indexer=True, na=False)
        frame_indx = muts['Protein'].str.match(
            pat='^p\\..*(?:\\*|(?:ins|del))', as_indexer=True, na=False)
        other_indx = ~(cnv_indx | point_indx | frame_indx)

        if any(cnv_indx):
            new_muts['CNV'] = muts.loc[cnv_indx, :]
        if any(point_indx):
            new_muts['Point'] = muts.loc[point_indx, :]
        if any(frame_indx):
            new_muts['Frame'] = muts.loc[frame_indx, :]
        if any(other_indx):
            new_muts['Other'] = muts.loc[other_indx, :]

        return new_muts

    @staticmethod
    def muts_location(muts):
        """Parses mutation according to protein location."""
        new_muts = {}

        loc_tbl = muts['Protein'].str.extract('(^p\\.[A-Z])([0-9]+)',
                                              expand=False)
        none_indx = pd.isnull(loc_tbl.ix[:, 1])
        loc_tbl.loc[none_indx, 1] = muts['Protein'][none_indx]

        for loc, grp in loc_tbl.groupby(by=1):
            new_muts[loc] = muts.ix[grp.index, :]

        return new_muts

    # .. functions for custom parsing of mutation levels ..
    @staticmethod
    def parse_base(muts, parse_lvl):
        """Removes trailing _Del and _Ins, merging insertions and deletions
           of the same type together.
        """
        new_lvl = parse_lvl + '_base'
        new_muts = muts.assign(**{new_lvl: muts.loc[:, parse_lvl]})
        new_muts = new_muts.replace(to_replace={new_lvl: {'_(Del|Ins)$': ''}},
                                    regex=True, inplace=False)

        return new_muts

    @staticmethod
    def parse_clust(muts, parse_lvl):
        """Clusters continuous mutation scores into discrete levels."""
        mshift = MeanShift(bandwidth=exp(-3))
        mshift.fit(pd.DataFrame(muts[parse_lvl]))
        clust_vec = [(parse_lvl + '_'
                      + str(round(mshift.cluster_centers_[x, 0], 2)))
                     for x in mshift.labels_]
        new_muts = muts
        new_muts[parse_lvl + '_clust'] = clust_vec

        return new_muts

    @staticmethod
    def parse_scores(muts, parse_lvl):
        return tuple(zip(muts['Sample'], pd.to_numeric(muts[parse_lvl])))

    def __new__(cls, muts, levels=('Gene', 'Form'), **kwargs):
        new_muts = cls.check_muts(muts, levels)
        if new_muts:
            return super(MuTree, cls).__new__(cls)
        else:
            return frozenset(muts['Sample'])

    def __init__(self, muts, levels=('Gene', 'Form'), **kwargs):
        """

        :param muts (pandas DataFrame):
        :param levels (tuple):
        :param kwargs:
        """

        if 'Sample' not in muts:
            raise ValueError("Mutation table must have a 'Sample' field!")

        if 'depth' in kwargs:
            self.depth = kwargs['depth']
        else:
            self.depth = 0

        # recursively builds the mutation hierarchy
        lvls_left = list(levels)
        self._child = {}
        rel_depth = 0

        while lvls_left and not self._child:
            cur_lvl = lvls_left.pop(0)
            splat_muts = self.split_muts(muts, cur_lvl)

            if splat_muts:
                self.mut_level = levels[rel_depth]
                if isinstance(splat_muts, tuple):
                    self._child = dict(splat_muts)

                else:
                    for nm, mut in splat_muts.items():
                        self._child[nm] = MuTree(mut, lvls_left,
                                                 depth=self.depth+1)
            else:
                rel_depth += 1

    def __iter__(self):
        """Allows iteration over mutation categories at the current level, or
           the samples at the current level if we are at a leaf node."""
        if isinstance(self._child, frozenset):
            return iter(self._child)
        else:
            return iter(self._child.items())

    def __getitem__(self, key):
        """Gets a particular category of mutations at the current level."""
        if not key:
            key_item = self

        elif isinstance(key, str):
            key_item = self._child[key]

        elif hasattr(key, '__getitem__'):
            sub_item = self._child[key[0]]

            if isinstance(sub_item, MuTree):
                key_item = sub_item[key[1:]]
            elif key[1:]:
                raise KeyError("Key has more levels than this MuTree!")
            else:
                key_item = sub_item

        else:
            raise TypeError("Unsupported key type " + type(key) + "!")

        return key_item

    def __str__(self):
        """Printing a MuTree shows each of the branches of the tree and
           the samples at the end of each branch."""
        new_str = self.mut_level

        for nm, mut in self:
            new_str += ' IS {}'.format(nm)

            if isinstance(mut, MuTree):
                new_str += (' AND ' + '\n'
                            + '\t' * (self.depth + 1) + str(mut))

            # if we have reached a root node, print the samples
            elif len(mut) > 8:
                    new_str += ': ({} samples)'.format(str(len(mut)))
            else:
                    new_str += ': {}'.format(
                        reduce(lambda x, y: '{},{}'.format(x, y), mut))

            new_str += ('\n' + '\t' * self.depth)
        new_str = gsub('\n$', '', new_str)

        return new_str

    def __len__(self):
        """Returns the number of unique samples this MuTree contains."""
        return len(self.get_samples())

    def get_levels(self):
        """Gets all the levels present in this tree and its children."""
        levels = {self.mut_level}

        for _, mut in self:
            if isinstance(mut, MuTree):
                levels |= set(mut.get_levels())

        return levels

    def get_samples(self):
        """Gets the set of unique samples contained within the tree."""
        samps = set()

        for nm, mut in self:
            if isinstance(mut, MuTree):
                samps |= mut.get_samples()
            elif isinstance(mut, frozenset):
                samps |= mut
            else:
                samps |= {nm}

        return samps

    def get_samp_count(self, samps):
        """Gets the number of branches of this tree each of the given
           samples appears in."""
        samp_count = {s:0 for s in samps}

        for _, mut in self:
            if isinstance(mut, MuTree):
                new_counts = mut.get_samp_count(samps)
                samp_count.update(
                    {s: (samp_count[s] + new_counts[s]) for s in samps})

            else:
                samp_count.update({s:(samp_count[s] + 1) for s in mut})

        return samp_count

    def get_overlap(self, mtype1, mtype2):
        """Gets the proportion of samples in one mtype that also fall under
           another, taking the maximum of the two possible mtype orders.

        Parameters
        ----------
        mtype1,mtype2 : MuTypes
            The mutation sets to be compared.

        Returns
        -------
        ov : float
            The ratio of overlap between the two given sets.
        """
        samps1 = mtype1.get_samples(self)
        samps2 = mtype2.get_samples(self)

        if len(samps1) and len(samps2):
            ovlp = float(len(samps1 & samps2))
            ov = max(ovlp / len(samps1), ovlp / len(samps2))

        else:
            ov = 0

        return ov

    def allkey(self, levels=None):
        """Gets the key corresponding to the MuType that contains all of the
           branches of the tree. A convenience function that makes it easier
           to list all of the possible branches present in the tree, and to
           instantiate MuType objects that correspond to all of the possible
           mutation types.

        Parameters
        ----------
        levels : tuple
            A list of levels corresponding to how far the output MuType
            should recurse.

        Returns
        -------
        new_key : dict
            A MuType key which can be used to instantiate
            a MuType object (see below).
        """
        if levels is None:
            levels = self.get_levels()
        new_lvls = set(levels) - {self.mut_level}

        if self.mut_level in levels:
            if '_scores' in self.mut_level:
                new_key = {(self.mut_level, 'Value'): None}

            else:
                new_key = {(self.mut_level, nm):
                           (mut.allkey(tuple(new_lvls))
                            if isinstance(mut, MuTree) and new_lvls
                            else None)
                           for nm, mut in self}

        else:
            new_key = reduce(
                lambda x,y: dict(
                    tuple(x.items()) + tuple(y.items())
                    + tuple((k, None) if x[k] is None
                            else (k, {**x[k], **y[k]})
                            for k in set(x) & set(y))),
                [mut.allkey(tuple(new_lvls))
                 if isinstance(mut, MuTree) and new_lvls
                 else {(self.mut_level, 'Value'): None}
                 if '_scores' in self.mut_level
                 else {(self.mut_level, nm): None}
                 for nm, mut in self]
                )

        return new_key

    def subsets(self, mtype=None, sub_levels=None, min_size=1):
        """Gets all MuTypes corresponding to one branch of the MuTree.

        Args:
            mtype (MuType), optional
                A set of mutations of which the returned MuTypes must be a
                subset. The default is to use all MuTypes within this MuTree.
            sub_levels (list of str), optional
                The levels of the leaf nodes of the returned MuTypes. The
                default is to use all levels of the MuTree.
            min_size (int), optional
                The minimum number of samples in each returned MuType. The
                default is not to do filtering based on MuType sample count.

        Returns:
            sub_mtypes (set of MuType)

        Examples:
            >>> # get all possible single-branch MuTypes
            >>> mtree.subsets()
            >>>
            >>> # get all possible MuTypes with at least five samples
            >>> mtree.subsets(min_size=5)
            >>>
            >>> # use different filters on the MuTypes returned for a given
            >>> # MuTree based on mutation type and mutation level
            >>> mtree.subsets(sub_levels=['Gene'])
                {MuType({('Gene', 'TP53'): None}),
                 MuType({('Gene', 'TTN'): None})}
            >>> mtree.subsets(sub_levels=['Gene', 'Type'])
                {MuType({('Gene', 'TP53'): {('Type', 'Point'): None}}),
                 MuType({('Gene', 'TP53'): {('Type', 'Frame'): None}}),
                 MuType({('Gene', 'TTN'): {('Type', 'Point'): None}})}
            >>> mtree.subsets(mtype=MuType({('Gene', 'TTN'): None}),
            >>>               sub_levels=['Gene', 'Type'])
                {MuType({('Gene', 'TTN'): {('Type', 'Point'): None}})}

        """
        sub_mtypes = set()

        # gets default values for filtering arguments
        if mtype is None:
            mtype = MuType(self.allkey())
        if sub_levels is None:
            sub_levels = self.get_levels()

        # finds the branches at the current mutation level that are a subset
        # of the given mutation type and have the minimum number of samples
        for nm, mut in self:
            for k, v in mtype:
                if k in nm and len(mut) >= min_size:

                    # returns the current branch if we are at one of the given
                    # mutation levels or at a leaf branch...
                    if self.mut_level in sub_levels or mut is None:
                        sub_mtypes |= {MuType({(self.mut_level, k): None})}

                    # ...otherwise, recurses into the children of the current
                    # branch that have at least one of the given levels
                    if (isinstance(mut, MuTree)
                            and set(sub_levels) & mut.get_levels()):
                        sub_mtypes |= set(
                            MuType({(self.mut_level, k): sub_mtype})
                            for sub_mtype in mut.subsets(
                                v, sub_levels, min_size)
                            )

        return sub_mtypes

    def combsets(self,
                 mtype=None, levels=None,
                 min_size=1, comb_sizes=(1,)):
        """Gets the MuTypes that are subsets of this tree and that contain
           at least the given number of samples and the given number of
           individual branches at the given hierarchy levels.

        Parameters
        ----------
        mtype : MuType
            A set of mutations whose subsets are to be obtained.

        levels : tuple
            The levels that the output sets are to contain.

        min_size : int
            The minimum number of samples each returned
            subset has to contain.

        comb_sizes : tuple of ints
            The number of individual branches each returned
            subset can contain.

        Returns
        -------
        csets : list
            A list of MuTypes satisfying the given criteria.

        """
        csets = {}
        all_subs = self.subsets(mtype, levels)

        for csize in comb_sizes:
            for kc in combn(all_subs, csize):
                new_set = reduce(lambda x, y: x | y, kc)

                if len(new_set.get_samples(self)) >= min_size:
                    csets |= {new_set}

        return csets

    def status(self, samples, mtype=None):
        """For a given set of samples and a MuType, finds if each sample
           has a mutation in the MuType in this tree.

        Parameters
        ----------
        samples : list
            A list of samples whose mutation status is to be retrieved.

        mtype : MuType, optional
            A set of mutations whose membership we want to test.
            The default is to check against any mutation
            contained in the tree.

        Returns
        -------
        S : list of bools
            For each input sample, whether or not it has a mutation in the
            given set.
        """
        if mtype is None:
            mtype = MuType(self.allkey())
        samp_list = mtype.get_samples(self)

        return np.array([s in samp_list for s in samples])


class MuType(object):
    """A particular type of mutation defined by annotation properties.

    A class corresponding to a subset of mutations defined through a hierarchy
    of properties. Used in conjunction with the above MuTree class to
    represent and navigate the space of possible mutation subsets.

    MuTypes are defined through a set key, which is a recursively structured
    dictionary of annotation property values of the form
        {(Level, Sub-Type1): (None or set_key), (Level, Sub-Type1): ...}

    Each item in the set key dictionary denotes a annotation property value
    contained within this mutation type. The key of an item is a 2-tuple
    with the first entry being a annotation hierarchy level (eg. 'Gene',
    'Form', 'Exon', etc.) and the second entry being a type or tuple of types
    available at this level (eg. 'KRAS', ('Missense_Mutation', 'Silent'),
    ('3/23', '6/13', '4/201'). The value of item can either be None, which
    means the mutation subtype contains all possible mutations with this
    property, or a set key to denote further subsetting of mutation types at
    more specific annotation property levels.

    All combinations of mutation subtypes within a MuType are defined as
    unions, that is, a MuType represents the abstract set of samples that
    has at least one of the mutation sub-types contained within it, as opposed
    to all of them.

    Arguments:
        set_key (dict): Defines the mutation sub-types included in this set.

    Attributes:
        cur_level (str): The mutation property level at the head of this set.

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

    def __init__(self, set_key):
        level = set(k for k, _ in set_key.keys())

        # gets the property hierarchy level of this mutation type after making
        # sure the set key is properly specified
        if len(level) > 1:
            raise ValueError("Improperly defined set key with multiple"
                             "mutation levels!")

        elif len(level) == 0:
            self.cur_level = None
        else:
            self.cur_level = tuple(level)[0]

        # gets the subsets of mutations defined at this level, and
        # their further subdivisions if they exist
        membs = [(k,) if isinstance(k, str) else k for _, k in set_key.keys()]
        children = {
            tuple(i for i in k):
            (ch if ch is None or isinstance(ch, MuType) else MuType(ch))
            for k, ch in zip(membs, set_key.values())
            }

        # merges subsets at this level if their children are the same:
        #   missense:None, frameshift:None => (missense,frameshift):None
        # or if they have the same keys:
        #   (missense, splice):M1, missense:M2, splice:M2
        #    => (missense, splice):(M1, M2)
        uniq_ch = set(children.values())
        uniq_vals = tuple((frozenset(i for j in
                                     [k for k, v in children.items()
                                      if v == ch] for i in j), ch)
                          for ch in uniq_ch)

        # adds the children nodes of this MuTree
        self.child = {}
        for val, ch in uniq_vals:

            if val in self.child:
                if ch is None or self.child[val] is None:
                    self.child[val] = None
                else:
                    self.child[val] |= ch

            else:
                self.child[val] = ch

    def __iter__(self):
        """Returns an expanded representation of the set structure."""
        return iter((l, v) for k, v in self.child.items() for l in k)

    def __eq__(self, other):
        """Two MuTypes are equal if and only if they have the same set
           of children MuTypes for the same subsets."""
        if isinstance(self, MuType) ^ isinstance(other, MuType):
            eq = False
        elif self.cur_level != other.cur_level:
            eq = False
        else:
            eq = (self.child == other.child)

        return eq

    def __repr__(self):
        """Shows the hierarchy of mutation properties contained
           within the MuType."""
        new_str = ''

        for k, v in self:
            if isinstance(k, str):
                new_str += self.cur_level + ' IS ' + k
            else:
                new_str += (self.cur_level + ' IS '
                            + reduce(lambda x, y: x + ' OR ' + y, k))

            if v is not None:
                new_str += ' AND ' + repr(v)
            new_str += ' OR '

        return gsub(' OR $', '', new_str)

    def __str__(self):
        """Gets a condensed label for the MuType."""
        new_str = ''

        for k, v in self:
            if v is None:
                new_str = new_str + k
            else:
                new_str = new_str + k + '-' + str(v)
            new_str = new_str + ', '

        return gsub(', $', '', new_str)

    def raw_key(self):
        "Returns the expanded key of a MuType."
        rmembs = reduce(lambda x,y: x|y, list(self.child.keys()))
        return {memb:reduce(lambda x,y: x|y,
                            [v for k,v in list(self.child.items())
                             if memb in k])
                for memb in rmembs}

    def __or__(self, other):
        """Returns the union of two MuTypes."""
        if not isinstance(other, MuType):
            return NotImplemented

        new_key = {}
        self_dict = dict(self)
        other_dict = dict(other)

        if self.cur_level == other.cur_level:
            for k in (self_dict.keys() - other_dict.keys()):
                new_key.update({(self.cur_level, k): self_dict[k]})
            for k in (other_dict.keys() - self_dict.keys()):
                new_key.update({(self.cur_level, k): other_dict[k]})

            for k in (self_dict.keys() & other_dict.keys()):
                if (self_dict[k] is None) or (other_dict[k] is None):
                    new_key.update({(self.cur_level, k): None})
                else:
                    new_key.update({
                        (self.cur_level, k): self_dict[k] | other_dict[k]})

        else:
            raise ValueError(
                "Cannot take the union of two MuTypes with mismatching "
                "mutation levels " + self.cur_level + " and "
                + other.cur_level + "!"
                )

        return MuType(new_key)

    def __and__(self, other):
        """Finds the intersection of two MuTypes."""
        if not isinstance(other, MuType):
            return NotImplemented

        new_key = {}
        self_dict = dict(self)
        other_dict = dict(other)

        if self.cur_level == other.cur_level:
            for k in (self_dict.keys() & other_dict.keys()):
                if self_dict[k] is None:
                    new_key.update({(self.cur_level, k): other_dict[k]})
                elif other_dict[k] is None:
                    new_key.update({(self.cur_level, k): self_dict[k]})
                else:
                    new_key.update({
                        (self.cur_level, k): self_dict[k] & other_dict[k]})

        else:
            raise ValueError("Cannot take the intersection of two MuTypes "
                             "with mismatching mutation levels "
                             + self.cur_level + " and "
                             + other.cur_level + "!")

        return MuType(new_key)

    def __ge__(self, other):
        """Checks if one MuType is a subset of the other."""
        if not isinstance(other, MuType):
            return NotImplemented

        self_dict = dict(self)
        other_dict = dict(other)

        if self.cur_level == other.cur_level:
            if self_dict.keys() >= other_dict.keys():
                for k in (self_dict.keys() & other_dict.keys()):
                    if self_dict[k] is not None:
                        if other_dict[k] is None:
                            return False
                        elif not (self_dict[k] >= other_dict[k]):
                            return False
                                
            else:
                return False
        else:
            return False

        return True

    def __gt__(self, other):
        """Checks if one MuType is a proper subset of the other."""
        if not isinstance(other, MuType):
            return NotImplemented

        self_dict = dict(self)
        other_dict = dict(other)

        if self.cur_level == other.cur_level:
            if self_dict.keys() > other_dict.keys():
                for k in (self_dict.keys() & other_dict.keys()):
                    if other_dict[k] is None:
                        return False
                    elif self_dict[k] is not None:
                        if not (self_dict[k] > other_dict[k]):
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

        new_key = {}
        self_dict = dict(self)
        other_dict = dict(other)

        if self.cur_level == other.cur_level:
            for k in self_dict.keys():
                if k in other_dict:
                    if other_dict[k] is not None:
                        if self_dict[k] is not None:
                            sub_val = self_dict[k] - other_dict[k]
                            if sub_val is not None:
                                new_key.update({(self.cur_level, k): sub_val})
                        else:
                            new_key.update(
                                {(self.cur_level, k): self_dict[k]})
                else:
                    new_key.update({(self.cur_level, k): self_dict[k]})

        else:
            raise ValueError("Cannot subtract MuType with mutation level "
                                 + other.cur_level + " from MuType with "
                                 + "mutation level " + self.cur_level + "!")

        if new_key:
            return MuType(new_key)
        else:
            return None

    def __hash__(self):
        """MuType hashes are defined in an analagous fashion to those of
           tuples, see for instance http://effbot.org/zone/python-hash.htm"""
        value = 0x163125

        for k,v in list(self.child.items()):
            value += eval(hex((int(value) * 1000007) & 0xFFFFFFFF)[:-1])
            value ^= hash(k) ^ hash(v)
            value ^= len(self.child)

        if value == -1:
            value = -2

        return value

    def get_levels(self):
        """Gets all the levels present in this type and its children."""
        levels = {self.cur_level}

        for _, v in self:
            if isinstance(v, MuType):
                levels |= set(v.get_levels())

        return levels

    def get_samples(self, mtree):
        """Gets the set of unique of samples contained within a particular
           branch or branches of the tree.

        Parameters
        ----------
        mtree : MuTree
            A set of samples organized according to the mutations they have.

        Returns
        -------
        samps : set
            The list of samples that have the specified type of mutations.
        """
        samps = set()

        if self.cur_level == mtree.mut_level:
            if '_scores' in self.cur_level:
                samps |= set(mtree._child.keys())

            else:
                for nm, mut in mtree:
                    for k, v in self:
                        if k == nm:
                            if isinstance(mut, frozenset):
                                samps |= mut
                            elif isinstance(mut, MuTree):
                                if v is None:
                                    samps |= mut.get_samples()
                                else:
                                    samps |= v.get_samples(mut)
                            else:
                                raise ValueError("get_samples error!")

        else:
            for _, mut in mtree:
                if isinstance(mut, MuTree):
                    samps |= self.get_samples(mut)

        return samps

    def invert(self, mtree):
        """Returns the mutation types not included in this set of types that
           are also in the given tree.
        """
        new_key = {}
        self_ch = self.raw_key()

        for k in (set(mtree.child.keys()) - set(self_ch.keys())):
            new_key[(self.cur_level, k)] = None

        for k in (set(mtree.child.keys()) & set(self_ch.keys())):
            if self_ch[k] is not None and isinstance(mtree.child[k], MuTree):
                new_key[(self.cur_level, k)] = self_ch[k].invert(
                    mtree.child[k])

        return MuType(new_key)

    def subkeys(self):
        """Gets all of the possible subsets of this MuType that contain
           exactly one of the leaf properties."""
        mkeys = []

        for k, v in list(self.child.items()):
            if v is None:
                mkeys += [{(self.cur_level, i): None} for i in k]
            else:
                mkeys += [{(self.cur_level, i): s}
                          for i in k for s in v.subkeys()]

        return mkeys
