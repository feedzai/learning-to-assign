def get_colorblind_cat_cmap(cmap='wong'):
    """
    Colorblind friendly color palettes.
    Source:https://davidmathlogic.com/colorblind/
    """
    supported_cmaps = {
        'wong': ['#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']
    }
    return supported_cmaps[cmap]
