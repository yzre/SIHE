# -*- encoding:utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib as mpl

PLTOPTS = {"color": "#33FFFF", "s": 15, "edgecolors": "none", "zorder": 5}
cmap = plt.get_cmap("jet")
norm = mpl.colors.Normalize(vmin=0.9, vmax=1.0)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
colors_tables = ['blue','orange','green','red','purple','brown','pink', 'gray', 'olive','cyan']
def c(x):
    return sm.to_rgba(x)


