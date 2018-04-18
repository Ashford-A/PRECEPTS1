
"""Pre-defined MuTypes used for testing.

"""

from ...features.mutations import MuType


# .. basic subtypes
basic = (
    MuType({('Gene', 'TP53'): None}),
    MuType({('Gene', 'TP53'): {('Form', 'Frame_Shift'): None}}),
    MuType({('Gene', ('TP53', 'TTN')): None}),
    MuType({('Gene', 'TP53'): None, ('Gene', 'TTN'): None}),
    MuType({('Gene', 'TP53'): None, ('Gene', ('TP53', 'TTN')): None}),
    MuType({('Gene', 'TP53'): {('Form', 'Frame_Shift'): None},
            ('Gene', ('TP53', 'TTN')): None}),
    )

# .. subtypes present in the ./mut_objs/muts_small.p set
small = (
    MuType({('Gene', 'TTN'): None}),
    MuType({('Gene', 'TTN'): {('Form', 'Missense_Mutation'): {
                ('Exon', ('326/363', '10/363')): None}}}),
    MuType({('Gene', 'CDH1'): None,
            ('Gene', 'TTN'): {('Form', 'Missense_Mutation'): {
                ('Exon', ('302/363', '10/363')): None}}}),
    MuType({('Form', 'Silent'): None}),
    MuType({('Gene', ('CDH1', 'TTN')): None})
    )

# .. subtypes which should never return any samples
blank = (
    MuType({('Gene', ('TTN', 'LOL5')): None}),
    MuType({('Gene', 'TTN'): {('Form', 'silly'): None}}),
    MuType({('Genie', 'Alladin'): None}),
    MuType({('Gene', 'TTN'): {('Gene', 'TTN2'): None}}),
    )

# .. paired subtypes which represent the same mutation sets
synonym = (
    MuType({('Gene', ('TTN', 'TP53')): None}),
    MuType({('Gene', 'TTN'): None, ('Gene', 'TP53'): None}),
    MuType({('Gene', 'TTN'): {
                ('Form', ('Silent', 'Splice')): {
                    ('Protein', ('p43T', 'p55H')): None}}}),
    MuType({('Gene', 'TTN'): {
                ('Form', 'Silent'): {('Protein', 'p43T'): None},
                ('Form', ('Splice', 'Silent')): {
                    ('Protein', 'p55H'): None},
                ('Form', 'Splice'): {('Protein', 'p43T'): None}
                }}),
    )

# .. subtypes unique to mutations in the TP53 gene
tp53 = (
    MuType({('Form', 'Splice_Site'): None}),
    MuType({('Exon', ('7/11', '8/11')): None}),
    MuType({('Protein', ('p.R158L', '.')): None}),
    )

# .. subtypes designed for use in testing binary operations
binary = (
    MuType({('Gene', ('BRAF', 'TP53')): {('Form', 'Splice_Site'): None},
            ('Gene', 'KRAS'): {
                ('Form', ('Splice_Site', 'Frame_Shift')): None}}),
    MuType({('Gene', 'TP53'): {
                ('Form', ('Splice_Site', 'Missense_Mutation')): None}}),
    MuType({('Gene', 'KRAS'): {
                ('Form', ('Splice_Site', 'Frame_Shift')): None},
            ('Gene', 'BRAF'): {('Form', 'Splice_Site'): None},
            ('Gene', 'TP53'): {
                ('Form', ('Splice_Site', 'Missense_Mutation')): None}}),
    MuType({('Gene', 'TP53'): {('Form', 'Splice_Site'): None}}),
    )

# .. subtypes for testing MuType sort order
sorting = (
    MuType({('Gene', ('ZFR3', 'TTN')): None}),
    MuType({('Gene', 'BRAF'): {('Form', 'Silent'): None}}),
    MuType({('Gene', 'BRAF'): {('Exon', ('1/21', '8/21')): None}}),
    MuType({('Gene', 'KRAS'): None}),
    MuType({('Gene', ('AR', 'ZFR3', 'KRAS')): None}),
    MuType({('Form', 'Missense_Mutation'): None}),
    MuType({('Gene', 'BRAF'): None}),
    MuType({('Gene', 'BRAF'): {('Exon', '8/21'): None}}),
    MuType({('Gene', ('BRAF', 'ZFR3', 'KRAS')): None}),
    MuType({('Gene', 'ZFR3'): None}),
    MuType({('Gene', 'BRAF'): {
                ('Form', 'Silent'): {('Exon', '8/21'): None}}}),
    MuType({('Gene', ('ZFR3', 'BRAF')): {('Form', 'Frame_Shift'): None}}),
    )

