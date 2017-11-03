
"""Loading and processing variant datasets.

This file contains functions and classes for loading, processing, and storing
mutations such as SNPs, indels, and frameshifts in formats suitable for use
in machine learning pipelines.

See Also:
    :module:`.utils`: Utilities common across many types of features.
    :module:`.copies`: Dealing with copy number alterations.

Author: Michal Grzadkowski <grzadkow@ohsu.edu>

"""

import numpy as np
import pandas as pd

import tarfile
import os
import glob

from io import BytesIO
import json

from re import sub as gsub
from math import exp
from ophion import Ophion

from operator import and_, or_
from functools import reduce

from itertools import combinations as combn
from itertools import permutations as perm
from itertools import product

from sklearn.cluster import MeanShift


# .. functions for loading mutation data from external data sources ..
def get_variants_mc3(syn):
    """Reads ICGC mutation data from the MC3 synapse file.

    Args:
        syn (Synapse): A logged-in synapseclient instance.

    Returns:
        muts (pandas DataFrame), shape = [n_mutations, mut_levels + 1]
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


def get_variants_firehose(cohort, data_dir):
    """Gets variant calls that have been downloaded from Firehose.

    Args:
        cohort (str): A TCGA cohort available in Broad Firehose.
        data_dir (str): A local directory where the data has been downloaded.

    Returns:

    Examples:

    """
    mut_tar = tarfile.open(glob.glob(os.path.join(
        data_dir, "stddata__2016_01_28", cohort, "20160128",
        "*Mutation_Packager_Oncotated_Calls.Level_3*tar.gz"
        ))[0])

    mut_list = []
    for mut_fl in mut_tar.getmembers():

        try:
            mut_tbl = pd.read_csv(
                BytesIO(mut_tar.extractfile(mut_fl).read()),
                sep='\t', skiprows=4, usecols=[0, 8, 15, 37, 41],
                names=['Gene', 'Form', 'Sample', 'Exon', 'Protein']
                )
            mut_list += [mut_tbl]

        except:
            print("Skipping mutations for {}".format(mut_fl))
        
    mut_data = pd.concat(mut_list)
    mut_data['Sample'] = ["-".join(x[:4])
                          for x in mut_data['Sample'].str.split('-')]

    return mut_data


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
    tree, thus representing a mutation sub-type such as 'TP53' for the Gene
    level, 'Missense_Mutation' for the Form level, 'R34K' for the Protein
    level, and so on.
    
    A node N* at the ith level of the tree has children nodes for each of
    the mutation types present at the (i+1)th annotation level for the samples
    also having mutations of type represented by N*. Thus in a tree
    containing the levels Gene, Form, and Exon, a node representing the ACT1
    gene will have a child representing missense mutations of ACT1, but only
    if at least one of the samples in the tree has this type of missense
    mutations. Similarly, this ACT1 - missense node may have children
    corresponding further sub-types of this mutation located on the 3rd, 5th,
    or 8th exon of ACT1.
    
    Every node in a MuTree is also a MuTree, except for the leaf nodes, which
    are frozensets of the samples which the mutation sub-type with all of the
    annotation level values of the parent nodes. Thus in the above example,
    the node representing the missense mutations of the ACT1 gene located on
    its 5th exon would simply be the samples with this mutation sub-type,
    since 'Exon' is the final annotation level contained in this MuTree.

    Levels can either be fields in the 'muts' DataFrame, in which case the
    tree will have a branch for each unique value in the field, or one of the
    keys of the MuTree.mut_fields object, in which case they will be defined
    by the corresponding MuType.muts_<level> method.

    Attributes:
        depth (int): How many mutation levels are above the tree
                     in the hierarchy.
        mut_level (str): The mutation annotation level described by the top
                         level of the tree.

    Args:
        muts (pandas DataFrame), shape = [n_muts, n_annot_fields]
            Input mutation data, each record is a mutation occurring in
            a sample to be included in the tree.
            Must contain a 'Sample' column.
        
        levels (:obj:`tuple` of :obj:`str`)
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

    @classmethod
    def split_muts(cls, muts, lvl_name):
        """Splits mutations into tree branches for a given level.

        Args:
            muts (pandas DataFrame), shape = [n_muts, n_annot_fields]
                A list of mutations to be split according to the given
                annotation level, where each row corresponds to a mutation
                in a particular sample. Must contain the annotation fields
                needed by the given level.
            lvl_name (str)
                An annotation level, must be either a column in the mutation
                dataframe, a parsed variation thereof, or a custom annotation
                level listed in `MuTree.mut_fields`.

        Returns:
            split_muts (:obj:`dict` of :obj:`pd.DataFrame`)
        """

        # level names have to consist of a base level name and an al
        # parsing label separated by an underscore
        lvl_info = lvl_name.split('_')
        if len(lvl_info) > 2:
            raise ValueError(
                "Invalid level name {} with more than two fields!".format(
                    lvl_name)
                )

        # if a parsing label is present, add the parsed level
        # to the table of mutations
        elif len(lvl_info) == 2:
            parse_lbl = lvl_info[1].lower()
            parse_fx = 'parse_{}'.format(parse_lbl)

            if parse_fx in cls.__dict__:
                muts = eval('cls.{}'.format(parse_fx))(muts, lvl_info[0])

            else:
                raise ValueError(
                    "Custom parse label {} must have a corresponding <{}> "
                    "method defined in {}!".format(
                        parse_lbl, parse_fx, cls.__name__)
                    )

        # splits mutations according to values of the specified level
        if isinstance(muts, tuple):
            if np.all(pd.isnull(val) for _, val in muts):
                split_muts = {}

            else:
                split_muts = muts

        elif lvl_name in muts:
            split_muts = dict(tuple(muts.groupby(lvl_name)))

        # if the specified level is not a column in the mutation table,
        # we assume it's a custom mutation level
        else:
            split_fx = 'muts_{}'.format(lvl_info[0].lower())

            if split_fx in cls.__dict__:
                split_muts = eval('cls.{}'.format(split_fx))(muts)

            else:
                raise ValueError(
                    "Custom mutation level {} must have a corresponding <{}> "
                    "method defined in {}!".format(
                        lvl_name, split_fx, cls.__name__)
                    )

        return split_muts

    """Functions for defining custom mutation levels.

    Args:
        muts (pandas DataFrame), shape = [n_muts, n_annot_fields]
            Mutations to be split according to the given level.
            Must contain a 'Sample' field as well as the fields defined in
            MuTree.mut_fields for each custom level.

    Returns:
        new_muts (:obj:`dict` of :obj:`pd.DataFrame`)
    """

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

    """Functions for custom parsing of mutation levels.

    Args:
        muts (pandas DataFrame), shape = [n_muts, n_annot_fields]
            Mutations whose properties are to be parsed.

    Returns:
        new_muts (pandas DataFrame), shape = [n_muts, n_annot_fields]
            The given list of mutations with the given mutation levels
            altered according to the corresponding parsing rule.
    """

    @staticmethod
    def parse_base(muts, parse_lvl):
        """Removes trailing _Del and _Ins, merging insertions and deletions
           of the same type together.
        """

        new_lvl = '{}_base'.format(parse_lvl)

        new_muts = muts.assign(**{new_lvl: muts.loc[:, parse_lvl]})
        new_muts.replace(to_replace={new_lvl: {'_(Del|Ins)$': ''}},
                         regex=True, inplace=True)

        return new_muts

    @staticmethod
    def parse_clust(muts, parse_lvl):
        """Clusters continuous mutation scores into discrete levels."""
        mshift = MeanShift(bandwidth=exp(-3))
        mshift.fit(pd.DataFrame(muts[parse_lvl]))

        clust_vec = [(parse_lvl + '_'
                      + str(round(mshift.cluster_centers_[x, 0], 2)))
                     for x in mshift.labels_]
        new_muts = muts.copy()
        new_muts[parse_lvl + '_clust'] = clust_vec

        return new_muts

    def __new__(cls, muts, levels=('Gene', 'Form'), **kwargs):
        """Given a list of mutations and a set of mutation levels, determines
           whether a mutation tree should be built, or a frozenset returned,
           presumably as a branch of another MuTree.

        """
        if 'Sample' not in muts:
            raise ValueError("Mutation table must have a 'Sample' field!")

        # initializes branch search variables
        muts_left = False
        lvls_left = list(levels)

        # look for a level at which MuTree branches can be sprouted until we
        # are either out of levels or we have found such a level
        while lvls_left and not muts_left:
            cur_lvl = lvls_left.pop(0).split('_')[0]

            # if the level is a field in the mutation DataFrame, check if any
            # mutations have non-null values...
            if cur_lvl in muts:
                muts_left = not np.all(pd.isnull(muts[cur_lvl]))

            # ...otherwise, check if the fields corresponding to the custom
            # level have any non-null values...
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

        # if we have found a level at which branches can be built,
        # continue with instantiating the MuTree...
        if muts_left:
            return super(MuTree, cls).__new__(cls)

        # ...otherwise, return a set of samples as a leaf node
        else:
            return frozenset(muts['Sample'])

    def __init__(self, muts, levels=('Gene', 'Form'), **kwargs):
        if 'depth' in kwargs:
            self.depth = kwargs['depth']
        else:
            self.depth = 0

        # intializes mutation hierarchy construction variables
        lvls_left = list(levels)
        self._child = {}
        rel_depth = 0

        # look for a mutation level at which we can create branches until we
        # have found such a level, note that we know such a level exists
        # because of the check performed in the __new__ method
        while lvls_left and not self._child:

            # get the split of the mutations given the current level
            cur_lvl = lvls_left.pop(0)
            splat_muts = self.split_muts(muts, cur_lvl)

            # if the mutations can be split, set the current mutation
            # level of the tree...
            if splat_muts:
                self.mut_level = levels[rel_depth]

                # ...and also set up the children nodes of the tree, which can
                # either all be frozensets corresponding to leaf nodes...
                if isinstance(splat_muts, tuple):
                    self._child = dict(splat_muts)

                # ...or a mixture of further MuTrees and leaf nodes
                else:
                    self._child = {nm: MuTree(mut, lvls_left,
                                              depth=self.depth+1)
                                   for nm, mut in splat_muts.items()}

            # if the mutations cannot be split at this level, move on to the
            # next level and keep track of how many levels we have skipped
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

    def sort_iter(self):

        if self.mut_level in ['Exon', 'Location']:
            return iter(sorted(
                [("0", branch) if lbl == '.' else (lbl, branch)
                 for lbl, branch in self._child.items()],
                key=lambda x: int(x[0].split('/')[0])
                ))

        else:
            return self.__iter__()

    def get_newick(self):
        """Get the Newick tree format representation of this MuTree."""
        newick_str = ''

        for nm, mut in self.sort_iter():

            if isinstance(mut, MuTree):
                newick_str += '(' + gsub(',$', '', mut.get_newick()) + ')'

            if nm == "0":
                newick_str += '{*none*},'
            else:
                newick_str += '{' + nm + '},'

        if self.depth == 0:
            newick_str = gsub(',$', '', newick_str) + ';' 

        return newick_str

    def get_levels(self):
        """Gets all the levels present in this tree and its children."""

        levels = {self.mut_level}

        for _, mut in self:
            if isinstance(mut, MuTree):
                levels |= mut.get_levels()

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

    def get_samp_count(self, samps=None):
        """How many unique branches is each sample located on?

        Returns:
            samp_count (dict): A count of branches for each given sample.

        """
        if samps is None:
            samps = self.get_samples()

        samp_count = {s: 0 for s in samps}
        for _, mut in self:

            if isinstance(mut, MuTree):
                new_counts = mut.get_samp_count(samps)

                samp_count.update({s: (samp_count[s] + new_counts[s])
                                   for s in samps})

            else:
                samp_count.update({s: (samp_count[s] + 1)
                                   for s in mut if s in samp_count})

        return samp_count

    def subtree(self, samps):
        """Modifies the MuTree in place so that it only has the given samples.

        Args:
            samps (list or set)

        Returns:
            self

        Examples:
            >>> # remove a sample from the tree
            >>> mtree = MuTree(...)
            >>> new_tree = mtree.subtree(mtree.get_samples() - {'TCGA-04'})

        """
        new_child = self._child.copy()
        for nm, mut in self:

            if isinstance(mut, MuTree):
                new_samps = mut.get_samples() & set(samps)
                if new_samps:
                    new_child[nm] = mut.subtree(new_samps)

            elif isinstance(mut, frozenset):
                new_samps = mut & frozenset(samps)
                if new_samps:
                    new_child[nm] = new_samps

            else:
                pass

        self._child = new_child
        return self

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

    def get_diff(self, mtype1, mtype2):
        """Gets the MuType of mutations in one MuType but not the other.

        Subtracts one set of mutations from another relative to this tree.
        Either given MuType may contain mutation types that are not present
        in any of the samples in the tree, but these types will be ignored
        for the purpose of identifying the difference set.

        Args:
            mtype1 (MuType): The mutation set to be subtracted from.
            mtype2 (MuType): The mutation set to be excluded.

        Returns:
            sub_mtype (MuType): The difference between the two given sets.

        """
        diff_key = {}

        # if the tree and the set to be subtracted from are at the same
        # mutation level, find the branches that are in both
        if mtype1.cur_level == self.mut_level:
            for (nm, branch), (_, btype) in filter(
                    lambda x: x[0][0] == x[1][0], product(self, mtype1)):

                # if we have reached a leaf branch in the tree, get
                # the MuType corresponding to this branch
                if isinstance(branch, frozenset):
                    use_btype = None

                # if the branch in the set includes all possible
                # sub-branches, enumerate these branches in the tree
                elif btype is None:
                    use_btype = MuType(
                        branch.allkey(levels=mtype2.get_levels()))

                # otherwise, use just the sub-branches present explicitly
                # listed in the set
                else:
                    use_btype = btype

                # add these sub-branches to the mutation set to be returned
                diff_key.update({(self.mut_level, nm): use_btype})

                # if the exclusion mutation set is also at the same mutation
                # level, find if it has the branch we are at
                if mtype2.cur_level == self.mut_level:
                    for sub_lbl, sub_btype in mtype2.subtype_list():
                        if sub_lbl == nm:

                            # delete the sub-branches if the branch to be
                            # excluded includes all sub-branches or is equal
                            # to the sub-branches to be subtracted from...
                            if sub_btype is None or sub_btype == use_btype:
                                del(diff_key[(self.mut_level, nm)])

                            # ...otherwise, recurse into the sub-branches in
                            # order to find the set difference
                            else:
                                diff_key.update(
                                    {(self.mut_level, nm): branch.get_diff(
                                        use_btype, sub_btype)}
                                    )

                elif mtype2.cur_level in self.get_levels():
                    diff_key.update({(self.mut_level, nm):
                                     branch.get_diff(use_btype, mtype2)})

                else:
                    diff_key.update({(self.mut_level, nm): None})

        elif mtype1.cur_level in self.get_levels():
            for nm, branch in self:
                diff_key.update({(self.mut_level, nm):
                                 branch.get_diff(mtype1, mtype2)})

        return MuType(diff_key)

    def allkey(self, levels=None):
        """Gets the key corresponding to the MuType with all the branches.

        A convenience function that makes it easier to list all of the
        possible branches present in the tree, and to instantiate MuType
        objects that correspond to all of the possible mutation types.

        Args:
            levels (list)

        Returns:
            dict

        """
        new_key = None

        # use all levels if no levels to filter on are provided
        if levels is None:
            levels = self.get_levels()

        if self.mut_level in levels:
            new_key = {
                (self.mut_level, nm): (
                    branch.allkey(branch.get_levels() & set(levels))
                    if (isinstance(branch, MuTree)
                        and branch.get_levels() & set(levels))
                    else None
                    )
                for nm, branch in self
                }

        elif set(levels) & self.get_levels():
            new_key = reduce(
                lambda x, y: dict(
                    tuple(x.items()) + tuple(y.items())
                    + tuple((k, None) if x[k] is None
                            else (k, {**x[k], **y[k]})
                            for k in set(x) & set(y))),
                [branch.allkey(branch.get_levels() & set(levels))
                 for nm, branch in self
                 if (isinstance(branch, MuTree)
                     and branch.get_levels() & set(levels))]
                )

        return new_key

    def rationalize(self, mtype):
        new_key = {}

        if self.mut_level == mtype.cur_level:
            in_stat = {nm: False for nm, _ in self}

            for (nm, branch), (_, btype) in filter(
                    lambda x: x[0][0] == x[1][0], product(self, mtype)):

                if btype is not None and isinstance(branch, MuTree):
                    new_key.update(
                        {(self.mut_level, nm): branch.rationalize(btype)})

                else:
                    new_key.update({(self.mut_level, nm): btype})

                if new_key[(self.mut_level, nm)] is None:
                    in_stat[nm] = True

            if all(in_stat.values()):
                new_key = None

        elif mtype.cur_level in self.get_levels():
            # TODO: consider mismatching MuType/MuTree levels
            pass

        if self.depth == 0:
            return MuType(new_key)
        else:
            return new_key

    def branchtypes(self, mtype=None, sub_levels=None, min_size=1):
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
            >>> mtree = MuTree(...)
            >>> mtree.branchtypes()
            >>>
            >>> # get all possible MuTypes with at least five samples
            >>> mtree.branchtypes(min_size=5)
            >>>
            >>> # use different filters on the MuTypes returned for a given
            >>> # MuTree based on mutation type and mutation level
            >>> mtree.branchtypes(sub_levels=['Gene'])
                {MuType({('Gene', 'TP53'): None}),
                 MuType({('Gene', 'TTN'): None})}
            >>> mtree.branchtypes(sub_levels=['Gene', 'Type'])
                {MuType({('Gene', 'TP53'): {('Type', 'Point'): None}}),
                 MuType({('Gene', 'TP53'): {('Type', 'Frame'): None}}),
                 MuType({('Gene', 'TTN'): {('Type', 'Point'): None}})}
            >>> mtree.branchtypes(mtype=MuType({('Gene', 'TTN'): None}),
            >>>               sub_levels=['Gene', 'Type'])
                {MuType({('Gene', 'TTN'): {('Type', 'Point'): None}})}

        """
        sub_mtypes = set()

        # gets default values for filtering arguments if they are missing
        if mtype is None:
            mtype = MuType(self.allkey())
        if sub_levels is None:
            sub_levels = self.get_levels()

        # finds the branches at the current mutation level that are a subset
        # of the given mutation type and have the minimum number of samples
        if self.mut_level in sub_levels:
            for (nm, branch), (_, btype) in filter(
                    lambda x: x[0][0] == x[1][0] and len(x[0][1]) >= min_size,
                    product(self, mtype.subtype_list())):

                # returns the current branch if we are at one of the given
                # mutation levels
                sub_mtypes.update({MuType({(self.mut_level, nm): None})})

                # ...otherwise, recurses into the children of the current
                # branch that have at least one of the given levels
                if (isinstance(branch, MuTree)
                        and set(sub_levels) & set(branch.get_levels())):
                
                    sub_mtypes |= set(
                        MuType({(self.mut_level, nm): rec_mtype})
                        for rec_mtype in branch.branchtypes(
                            btype, sub_levels, min_size)
                        )

        # otherwise, if we are not at one of the mutation levels to
        # retrieve MuTypes at but are at the same level in the
        # subset MuType and MuTree...
        elif mtype.cur_level == self.mut_level:
            new_key = {}

            for lbls, btype in mtype.child_iter():
                for nm, branch in self:

                    if nm in lbls:
                        rec_mtypes = branch.branchtypes(
                            btype, sub_levels, min_size=1)

                        for rec_mtype in rec_mtypes:
                            if rec_mtype in new_key:
                                new_key[rec_mtype] |= {nm}
                            else:
                                new_key[rec_mtype] = {nm}

            for rec_mtype, nms in new_key.items():
                rec_samp_count = len(rec_mtype.get_samples(self)
                                     & mtype.get_samples(self))

                if rec_samp_count >= min_size:
                    sub_mtypes.update({
                        MuType({(self.mut_level, tuple(nms)): rec_mtype})})

        else:
            recurse_mtypes = reduce(
                or_,
                [branch.branchtypes(mtype, sub_levels, min_size=1)
                 for _, branch in self],
                set()
                )

            sub_mtypes |= set(filter(
                lambda x: (len(x.get_samples(self) & mtype.get_samples(self))
                           >= min_size),
                recurse_mtypes
                ))

        return sub_mtypes

    def windowtypes(self,
                    mtype, sub_level='Exon', min_samps=1,
                    wind_width=2, wind_overlap=0, abs_windows=False):
        """Gets MuTypes corresponding to tiled intervals along a continuum.

        """

        if sub_level not in self.get_levels():
            raise ValueError("Sub-mutation level {} is not stored in this"
                             "MuTree!".format(sub_level))

        if wind_overlap >= wind_width:
            raise ValueError(
                "Sub-type window overlap must be smaller than window size!")

        # gets default values for filtering arguments if they are missing
        if mtype is None:
            mtype = MuType(self.allkey())

        if self.mut_level == sub_level:
            sorted_types = tuple(sorted(
                ["0" if x == '.' else x for x in self._child.keys()],
                key=lambda lbl: int(lbl.split('/')[0])))

        elif mtype.cur_level == self.mut_level:
            rec_mtypes = set()

            for lbls, btype in mtype.child_iter():
                for nm, branch in self:

                    if nm in lbls:
                        rec_mtypes |= branch.branchtypes(
                            btype, [sub_level], min_size=1)

            sorted_types = tuple(sorted(
                ["0" if x == '.' else x for x in [str(mtp).split('-')[-1] for mtp
                                                in rec_mtypes]],
                key=lambda lbl: int(lbl.split('/')[0])
                ))

        else:
            sorted_types = tuple()

        if abs_windows:
            mtype_ind = [int(lbl.split('/')[0]) for lbl in sorted_types]
            wind_ind = tuple(range(0, mtype_ind[-1] + 1,
                                   wind_width - wind_overlap))
            block_types = [
                tuple(sorted_types[i] for i, ind in enumerate(mtype_ind)
                      if wind_ind[j] <= ind <= (wind_ind[j] + wind_width))
                for j in range(len(wind_ind) - 1)
                ]

            wind_mtypes = set(mtype & MuType({(sub_level, bk_type): None})
                              for bk_type in block_types)

        else:
            wind_mtypes = set(
                mtype & MuType({
                    (sub_level,
                     sorted_types[i:(i + wind_width)]): None})
                for i in range(0, len(sorted_types) - wind_width + 1,
                               wind_width - wind_overlap)
            )

        wind_mtypes = set(mtype for mtype in wind_mtypes
                          if len(mtype.get_samples(self)) >= min_samps)

        return wind_mtypes

    def combtypes(self,
                  mtype=None, sub_levels=None, comb_sizes=(1, 2),
                  min_type_size=10, min_branch_size='auto'):
        """Gets all MuTypes that combine multiple branches of the tree.

        Args:
            mtype (MuType), optional
                A set of mutations of which the returned MuTypes must be a
                subset. The default is to use all MuTypes within this MuTree.
            sub_levels (list of str), optional
                The levels of the leaf nodes of the returned MuTypes. The
                default is to use all levels of the MuTree.
            comb_sizes (list of int), optional
                The number of branches that each returned MyType can combine.
                The default is to consider combinations of up to two branches.
            min_type_size (int), optional
                The minimum number of samples in each returned MuType. The
                default is each returned MuType having at least ten samples.
            min_branch_size (:obj:`int` or :obj:`str`), optional
                The minimum number of samples in each of the MuTypes combined
                to compose each of the returned MuTypes. The default is to
                divide min_type_size by the number of MuTypes in each returned
                combination MuType.

        Returns:
            comb_mtypes (:obj:`set` of :obj:`MuType`)

        Examples:
            >>> # get all possible MuTypes that combine three branches
            >>> mtree = MuTree(...)
            >>> mtree.combtypes(comb_sizes=(3,))
            >>>
            >>> # get all possible MuTypes that combine two 'Type' branches
            >>> # that have at least twenty samples in this tree
            >>> mtree.combtypes(min_size=20, sub_levels=['Type'])

        """
        comb_mtypes = set()

        if not isinstance(min_branch_size, str):
            branch_mtypes = self.branchtypes(
                mtype, sub_levels, min_size=min_branch_size)

        for csize in comb_sizes:
            if min_branch_size == 'auto':
                branch_mtypes = self.branchtypes(
                    mtype, sub_levels, min_size=(min_type_size / csize))

            if branch_mtypes:
                for mtype_combs in combn(branch_mtypes, csize):

                    if (csize == 1
                            or all([not (mtype1.is_supertype(mtype2)
                                         or mtype2.is_supertype(mtype1))
                                    for mtype1, mtype2
                                    in combn(mtype_combs, 2)])):
                        new_mtype = reduce(or_, mtype_combs)

                        if (min_branch_size == 'auto'
                                or (len(new_mtype.get_samples(self))
                                    >= min_type_size)):
                            comb_mtypes |= {new_mtype}

        return comb_mtypes

    def treetypes(self, mtype=None, sub_levels=None, min_size=1):
        """Get all MuTypes that combine any number of sub-branches
           of a mutation level.

        """
        tree_mtypes = set()

        if mtype is None:
            mtype = MuType(self.allkey())
        if sub_levels is None:
            sub_levels = self.get_levels()

        if self.mut_level in sub_levels:
            if len(self._child) > 1 or (len(self._child) == 1
                                        and self.mut_level == sub_levels[0]):

                tree_mtypes |= self.combtypes(
                    mtype=mtype, sub_levels=[self.mut_level],
                    comb_sizes=range(1, max(2, len(self._child))),
                    min_size=min_size
                    )

            for (nm, branch), (_, btype) in filter(
                    lambda x: x[0][0] == x[1][0] and len(x[0][1]) > min_size,
                    product(self, mtype)
                    ):

                if (isinstance(branch, MuTree)
                        and set(sub_levels) & set(branch.get_levels())):
                    tree_mtypes |= set(
                        MuType({(self.mut_level, nm): tree_mtype})
                        for tree_mtype in branch.treetypes(
                            btype, sub_levels, min_size)
                        )

        else:
            tree_mtypes |= reduce(
                or_,
                [branch.treetypes(btype, sub_levels, min_size)
                 for (nm, branch), (lbl, btype) in product(self, mtype)
                 if (isinstance(branch, MuTree)
                     and nm == lbl and len(branch) > min_size
                     and set(sub_levels) & set(branch.get_levels()))],
                set()
                )

        return tree_mtypes

    def status(self, samples, mtype=None):
        """Finds if each sample has a mutation of this type in the tree.

        Args:
            samples (:obj:`list` of :obj:`str`)
                Which samples' mutation status is to be retrieved.
            mtype (MuType, optional)
                A set of mutations whose membership we want to test.
                The default is to check against any mutation
                contained in the tree.

        Returns:
            stat_list (:obj:`list` of :obj:`bool`)
                For each input sample, whether or not it has a mutation
                in the given set.

        """
        if mtype is None:
            mtype = MuType(self.allkey())

        samp_list = mtype.get_samples(self)
        stat_list = [s in samp_list for s in sorted(samples)]

        return stat_list


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
        type_dict (dict): The mutation sub-types included in this type.

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
            (lbl,
             (sub_type if sub_type is None or isinstance(sub_type, MuType)
              else MuType(sub_type)))
            for lbls, sub_type in zip(level_lbls, type_dict.values())
            if not (isinstance(sub_type, MuType) and sub_type.is_empty())
            for lbl in lbls
            ]

        full_dict = {}
        for lbl, sub_type in full_pairs:

            if lbl in full_dict:
                if sub_type is None or full_dict[lbl] is None:
                    full_dict[lbl] = None

                else:
                    full_dict[lbl] |= sub_type

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

            new_str += ', OR '

        return gsub(', OR $', '', new_str)

    def __str__(self):
        """Gets a condensed label for the MuType."""
        new_str = ''
        self_iter = sorted(self._child.items(), key=lambda x: list(x[0]))

        # if there aren't too many types to list at this mutation level...
        if len(self_iter) <= 10:

            # ...iterate over the types, grouping together those with the
            # same children to produce a more concise label
            for lbls, tp in self_iter:

                if len(lbls) > 1:
                    new_str += "({})".format('|'.join(sorted(lbls)))

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
            mtree (MuTree): A hierarchy of mutations present in a cohort.

        Returns:
            samps (set): The samples in the MuTree that have the mutation(s)
                         specified by this MuType.
                         .
        """
        if not isinstance(mtree, MuTree):
            raise TypeError("Can't retrieve samples from something that is "
                            "not a MuTree!")

        # if this MuType has the same mutation level as the MuTree...
        samps = set()
        if self.cur_level == mtree.mut_level:

            # ...find the mutation entries in the MuTree that match the
            # mutation entries in the MuType
            for (nm, mut), (lbl, tp) in product(mtree, self.subtype_list()):
                if lbl == nm:
                    
                    if isinstance(mut, frozenset):
                            samps |= mut
                    elif isinstance(mut, MuTree):
                        if tp is None:
                            samps |= mut.get_samples()
                        else:
                            samps |= tp.get_samples(mut)
                    else:
                        raise ValueError("get_samples error!")

        else:
            for _, mut in mtree:
                if (isinstance(mut, MuTree)
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

        for lbls, tp in list(self._child.items()):
            if tp is None:
                mkeys += [{(self.cur_level, lbl): None} for lbl in lbls]
            else:
                mkeys += [{(self.cur_level, lbl): sub_tp}
                          for lbl in lbls for sub_tp in tp.subkeys()]

        return mkeys


class MutComb(object):
    """A class corresponding to the presence of multiple mutation sub-types.

    Args:
        mtypes (:obj:`list` of :obj:`MuType`)

    Examples:
        >>> # create an object representing samples that have both a TTN
        >>> # mutation and a KRAS missense mutation
        >>>
        >>> from HetMan.features.variants import MuType
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
