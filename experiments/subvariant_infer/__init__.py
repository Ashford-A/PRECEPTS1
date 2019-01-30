
from .data_dirs import expr_dir, copy_dir, syn_root, domain_dir, annot_file
from .setup_infer import Mcomb
from dryadic.features.mutations import MuType


# define major mutation types
variant_mtypes = (
    ('Loss', MuType({('Scale', 'Copy'): {(
        'Copy', ('ShalDel', 'DeepDel')): None}})),
    ('Point', MuType({('Scale', 'Point'): None})),
    ('Gain', MuType({('Scale', 'Copy'): {(
        'Copy', ('ShalGain', 'DeepGain')): None}}))
    )

# define plotting colours for the major mutation types
variant_clrs = {'Point': "#0D29FF", 'Gain': "#6AC500", 'Loss': "#BB0048"}

variant_mcombs = (
    ('Point+Loss', Mcomb(dict(variant_mtypes)['Point'],
                         dict(variant_mtypes)['Loss'])),
    ('Point+Gain', Mcomb(dict(variant_mtypes)['Point'],
                         dict(variant_mtypes)['Gain']))
    )

mcomb_clrs = {'Point+Loss': "#7C30B0", 'Point+Gain': "#25A497"}


__all__ = ['expr_dir', 'copy_dir', 'syn_root', 'domain_dir', 'annot_file']

