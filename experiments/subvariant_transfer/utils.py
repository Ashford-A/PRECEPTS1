
from HetMan.experiments.subvariant_infer.setup_infer import Mcomb, ExMcomb


def get_form(mtype):
    use_type = mtype.subtype_list()[0][1]
 
    if isinstance(use_type, ExMcomb) or isinstance(use_type, Mcomb):
        if len(use_type.mtypes) == 1:
            use_subtype = tuple(use_type.mtypes)[0]
            mtype_lvls = use_subtype.get_sorted_levels()[1:]
        else:
            mtype_lvls = None
 
    else:
        use_subtype = use_type
        mtype_lvls = use_type.get_sorted_levels()[1:]
 
    if mtype_lvls == ('Copy', ):
        copy_type = use_subtype.subtype_list()[0][1].subtype_list()[0][0]

        if copy_type == 'DeepGain':
            mtype_form = 'Gain'
        elif copy_type == 'DeepDel':
            mtype_form = 'Loss'
        else:
            mtype_form = 'Other'

    else:
        mtype_form = 'Point'

    return mtype_form

