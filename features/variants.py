
"""
HetMan (Heterogeneity Manifold)
Classification of mutation sub-types using expression data.
This file contains classes for representing and storing variants.
See .copies for code dealing with copy number alterations.
"""

# Author: Michal Grzadkowski <grzadkow@ohsu.edu>

import numpy as np
import pandas as pd

from re import sub as gsub
from math import exp
import json
from ophion import Ophion

from functools import reduce
from itertools import combinations as combn

from sklearn.cluster import MeanShift


# .. functions for loading mutation data from external data sources ..
def get_variants_mc3(syn):
    """Reads ICGC mutation data from the MC3 synapse file.

    Parameters
    ----------
    syn : object
        An instance of Synapse that has already been logged into.

    Returns
    -------
    muts : ndarray, shape (n_mutations, mut_levels+1)
        A mutation array, with a row for each mutation appearing in an
        individual sample.
    """

    # gets data from Synapse, figures out which columns to use
    mc3 = syn.get('syn7824274')
    use_cols = [0, 8, 15, 36, 38, 72]
    use_names = ['Gene', 'Form', 'Sample', 'Protein', 'Exon', 'PolyPhen']

    # imports data into a DataFrame, parses TCGA sample barcodes
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
    mut_list = {samp:{} for samp in sample_list}
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
    """A class corresponding to a hierarchy of mutation types
       present in a set of samples.
       
    Parameters
    ----------
    muts : pandas DataFrame, shape (n_muts,)
        Input mutation data, each record is a mutation occurring in a sample.
        Must contain a 'Sample' column.
        
    levels : tuple
        A list of mutation levels to be included in the tree.
        All sub-trees will have list the same set of levels regardless of
        their depth in the hierarchy.

        Levels can either be fields in the 'muts' DataFrame, in which case
        the tree will have a branch for each unique value in the field, or
        one of the keys of the MuTree.mut_fields object, in which case they
        will be defined by the corresponding MuType.muts_<level> method.

        Mutation trees can either have other mutation trees as children,
        corresponding to lower levels in the hierarchy, or have lists of
        individual samples as children if they are at the very bottom of the
        hierarchy which are stored as frozensets in the case of discrete
        mutation types and dicts in the case of continuous mutations.
    """

    # mapping between mutation fields and custom mutation levels
    mut_fields = {
        'Type': ['Gene', 'Form', 'Protein']
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
                muts_left = not all(pd.isnull(muts[cur_lvl]))

            elif cur_lvl in cls.mut_fields:
                if not all([x in muts for x in cls.mut_fields[cur_lvl]]):
                    raise ValueError("For mutation level " + cur_lvl + ", "
                                     + str(cls.mut_fields[cur_lvl])
                                     + " need to be provided as fields.")
                else:
                    muts_left = not all(pd.isnull(
                        muts.loc[:, cls.mut_fields[cur_lvl]]))

            else:
                raise ValueError("Unknown mutation level " + cur_lvl
                                     + " which is not in the given mutation "
                                     "data frame nor a custom-defined level!")

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
                raise ValueError("Custom parse label " + parse_lbl
                                     + " must have a corresponding <"
                                     + parse_fx + "> method defined in "
                                     + cls.__name__ + "!")

        # splits mutations according to values of the specified level
        if isinstance(muts, tuple):
            if all(pd.isnull(val) for _, val in muts):
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
                      + str(round(mshift.cluster_centers_[x,0], 2)))
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
        if 'Sample' not in muts:
            raise ValueError("Mutations must have a 'Sample' field!")
        if 'depth' in kwargs:
            depth = kwargs['depth']
        else:
            depth = 0
        self.depth = depth

        # recursively builds the mutation hierarchy
        lvls_left = list(levels)
        self.child = {}
        rel_depth = 0

        while lvls_left and not self.child:
            cur_lvl = lvls_left.pop(0)
            splat_muts = self.split_muts(muts, cur_lvl)

            if splat_muts:
                self.cur_level = levels[rel_depth]
                if isinstance(splat_muts, tuple):
                    self.child = dict(splat_muts)

                else:
                    for nm, mut in splat_muts.items():
                        self.child[nm] = MuTree(mut, lvls_left,
                                                depth=self.depth+1)
            else:
                rel_depth += 1

    def __iter__(self):
        """Allows iteration over mutation categories at the current level."""
        if isinstance(self.child, frozenset):
            return iter(self.child)
        else:
            return iter(self.child.items())

    def __getitem__(self, key):
        """Gets a particular category of mutations at the current level."""
        if not key:
            key_item = self
        elif isinstance(key, str):
            key_item = self.child[key]
        elif hasattr(key, '__getitem__'):
            sub_item = self.child[key[0]]
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
        new_str = self.cur_level

        # if the current level is continuous, print summary statistics
        if '_scores' in self.cur_level:
            if len(self) > 8:
                score_dict = np.percentile(tuple(self.get_scores().values()),
                                           [0, 25, 50, 75, 100])
                new_str = (new_str + ": {} samples with score distribution "
                           "Min({:05.4f}) 1Q({:05.4f}) Med({:05.4f}) "
                           "3Q({:05.4f}) Max({:05.4f})".format(
                               len(self), *score_dict))
            else:
                new_str = new_str + ': ' + str(self.get_scores())

        # otherwise, iterate over the branches, recursing when necessary
        else:
            for nm, mut in self:
                new_str = new_str + ' IS ' + nm
                if isinstance(mut, MuTree):
                    new_str = (new_str + ' AND '
                               + '\n' + '\t'*(self.depth+1) + str(mut))

                # if we have reached a root node, print the samples
                else:
                    if not hasattr(mut, '__len__'):
                        new_str = new_str + str(round(mut, 2))
                    elif len(mut) > 10:
                        new_str = (new_str
                                   + ': (' + str(len(mut)) + ' samples)')
                    elif isinstance(mut, frozenset):
                        new_str = (new_str + ': '
                                   + reduce(lambda x,y: x + ',' + y, mut))
                new_str = new_str + '\n' + '\t'*self.depth
            new_str = gsub('\n$', '', new_str)

        return new_str

    def __len__(self):
        """Returns the number of unique samples this MuTree contains."""
        return len(self.get_samples())

    def get_levels(self):
        """Gets all the levels present in this tree and its children."""
        levels = set([self.cur_level])

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

        for v in list(self.child.values()):
            if isinstance(v, MuTree):
                new_counts = v.get_samp_count(samps)
                samp_count.update(
                    {s:(samp_count[s] + new_counts[s]) for s in samps})
            else:
                samp_count.update({s:(samp_count[s] + 1) for s in v})

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
        samps1 = self.get_samples(mtype1)
        samps2 = self.get_samples(mtype2)
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
        new_lvls = set(levels) - set([self.cur_level])

        if self.cur_level in levels:
            if '_scores' in self.cur_level:
                new_key = {(self.cur_level, 'Value'): None}

            else:
                new_key = {(self.cur_level, nm):
                           (mut.allkey(new_lvls)
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
                [mut.allkey(new_lvls) if isinstance(mut, MuTree) and new_lvls
                 else {(self.cur_level, 'Value'): None}
                 if '_scores' in self.cur_level
                 else {(self.cur_level, nm): None}
                 for nm, mut in self]
                )

        return new_key

    def subsets(self, mtype=None, levels=None):
        """Gets all of the MuTypes corresponding to exactly one of the
           branches of the tree within the given mutation set and at the
           given mutation levels.

        Parameters
        ----------
        mtype : MuType, optional
            A set of mutations whose sub-branches are to be obtained.

        levels : tuple, optional
            A list of levels where the sub-branches are to be located.

        Returns
        -------
        mtypes : list
            A list of MuTypes, each corresponding to one of the
            branches of the tree.
        """
        if mtype is None:
            mtype = MuType(self.allkey(levels))
        if levels is None:
            levels = self.get_levels()
        mtypes = []

        if self.cur_level in levels:
            for nm, mut in self:
                for k, v in mtype:
                    if k in nm:
                        new_lvls = list(set(levels) - {[self.cur_level]})
                        if isinstance(mut, MuTree) and len(new_lvls) > 0:
                            mtypes += [MuType({(self.cur_level, k): s})
                                       for s in mut.subsets(v, new_lvls)]
                        else:
                            mtypes += [MuType({(self.cur_level, k): None})]

        else:
            mtypes += [mut.subsets(mtype, levels) for _, mut in self
                       if isinstance(mut, MuTree)]

        return mtypes

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
        all_subs = self.subsets(mtype, levels)
        csets = []
        for csize in comb_sizes:
            for kc in combn(all_subs, csize):
                new_set = reduce(lambda x, y: x | y, kc)
                if len(new_set.get_samples(self)) >= min_size:
                    csets += [new_set]

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

        return [s in samp_list for s in samples]


class MuType(object):
    """A class corresponding to a subset of mutations defined through
       hierarchy of properties. Used in conjunction with the above MuTree
       class to navigate the space of possible mutation subsets.

    Parameters
    ----------
    set_key : dict
        Define the mutation sub-types that are to be included in this set.
        Takes the form {(Level,Sub-Type):None or set_key, ...}.

        A value of None denotes all of the samples with the given sub-type of
        mutation at the given level, otherwise another set-key which defines a
        further subset of mutations contained within the given sub-type.
        Sub-Type can consist of multiple values, in which case the
        corresponding value applies to all of the included sub-types.

        i.e. {('Gene','TP53'):None} is the subset containing any mutation
        of the TP53 gene.
        {('Gene','BRAF'):{('Conseq',('missense','frameshift')):None}} contains
        the mutations of BRAF that result in a missense variation or a shift
        of the reading frame.

        As with MuTrees, MuTypes are constructed recursively, and so each
        value in a set key is used to create another MuType, unless it is None
        signifying a leaf node in the hierarchy.

    Attributes
    ----------
    cur_level : str
        The mutation level at the head of this mutation set.
    """

    def __init__(self, set_key):
        # gets the mutation hierarchy level of this set, makes sure
        # the key is properly specified
        level = set(k for k,_ in list(set_key.keys()))
        if len(level) > 1:
            raise ValueError(
                "improperly defined MuType key (multiple mutation levels)")
        if level:
            self.cur_level = tuple(level)[0]
        else:
            self.cur_level = None

        # gets the subsets of mutations defined at this level, and
        # their further subdivisions if they exist
        membs = [(k,) if isinstance(k, str) else k
                 for _,k in list(set_key.keys())]
        children = {
            tuple(i for i in k):
            (ch if ch is None or isinstance(ch, MuType) else MuType(ch))
            for k,ch in zip(membs, set_key.values())
            }

        # merges subsets at this level if their children are the same:
        #   missense:None, frameshift:None => (missense,frameshift):None
        # or if they have the same keys:
        #   (missense, splice):M1, missense:M2, splice:M2
        #    => (missense, splice):(M1, M2)
        uniq_ch = set(children.values())
        uniq_vals = tuple((frozenset(i for j in
                              [k for k,v in children.items() if v == ch]
                              for i in j), ch) for ch in uniq_ch)

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
                            + reduce(lambda x,y: x + ' OR ' + y, k))

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
        levels = set([self.cur_level])

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

        if self.cur_level == mtree.cur_level:
            if '_scores' in self.cur_level:
                samps |= set(mtree.child.keys())

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

