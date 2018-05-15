
from .isolate_cohort_mutypes import load_output as iso_output
from .cross_cohort_mutypes import load_output as cross_output

from matplotlib.colors import LinearSegmentedColormap


cdict = {
    'red': ((0.0,  2.0/3, 2.0/3),
            (1.0/3,  1.0, 1.0),
            (0.5,  0.224, 0.224),
            (2.0/3,  0.0, 0.0),
            (1.0,  0.392, 0.392)),
    
    'green': ((0.0,  0.329, 0.329),
              (1.0/3,  1.0, 1.0),
              (0.5,  0.191, 0.191),
              (2.0/3,  0.0, 0.0),
              (1.0,  0.729, 0.729)),
 
    'blue': ((0.0,  0.31, 0.31),
             (1.0/3,  1.0, 1.0),
             (0.5,  0.451, 0.451),
             (2.0/3,  0.0, 0.0),
             (1.0,  0.416, 0.416))
    }

simil_cmap = LinearSegmentedColormap('SimilCmap', cdict)


__all__ = ['iso_output', 'cross_output', 'simil_cmap']

