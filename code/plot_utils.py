import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np

COLOUR_PALETTE_3 = [
    "#C89F03",  # warm gold (anchor)
    "#141644",  # deep navy (anchor)
    "#2E8B7A",  # teal-green contrast
]
GRADIENT_PALETTE_3 = []
for c in COLOUR_PALETTE_3:
    grad = sns.light_palette(c, n_colors=5, reverse=False)
    GRADIENT_PALETTE_3.extend(grad)
SEQUENTIAL_CMAP_3 = sns.light_palette("#2E8B7A", as_cmap=True)
DIVERGING_CMAP_3 = sns.diverging_palette(
    260, 180,  # hue angles
    s=80, l=55,
    as_cmap=True
)
BINARY_PALETTE_3 = sns.diverging_palette(
    260, 180,  # hue angles
    s=80, l=55,
    n=2,
    as_cmap=False
)

COLOUR_PALETTE_4 = [
    "#C89F03",  # warm gold
    "#141644",  # deep navy
    "#D1495B",  # coral red
    "#17BEBB",  # bright teal
]
GRADIENT_PALETTE_4 = []
for c in COLOUR_PALETTE_4:
    grad = sns.light_palette(c, n_colors=5, reverse=False)
    GRADIENT_PALETTE_4.extend(grad)
SEQUENTIAL_CMAP_4 = sns.light_palette("#D1495B", as_cmap=True)
DIVERGING_CMAP_4 = sns.diverging_palette(
    10, 220,
    s=75, l=50,
    as_cmap=True
)
BINARY_PALETTE_4 = sns.diverging_palette(
    10, 220,
    s=75, l=50,
    n=2,
    as_cmap=False
)

COLOUR_PALETTE_5 = [
    "#C89F03",  # warm gold
    "#141644",  # deep navy
    "#D1495B",  # coral red
    "#17BEBB",  # bright teal
    "#88B04B",  # moss green
]
GRADIENT_PALETTE_5 = []
for c in COLOUR_PALETTE_5:
    grad = sns.light_palette(c, n_colors=5, reverse=False)
    GRADIENT_PALETTE_5.extend(grad)
SEQUENTIAL_CMAP_5 = sns.light_palette("#88B04B", as_cmap=True)
DIVERGING_CMAP_5 = sns.diverging_palette(
    140, 10,
    s=80, l=50,
    as_cmap=True
)
BINARY_PALETTE_5 = sns.diverging_palette(
    140, 10,
    s=80, l=50,
    n=2,
    as_cmap=False
)

COLOUR_PALETTE_8 = [
    "#C89F03",  # warm gold
    "#141644",  # deep navy
    "#D1495B",  # coral red
    "#17BEBB",  # bright teal
    "#88B04B",  # moss green
    "#F29E4C",  # soft orange
    "#6A5FCD",  # muted purple
    "#999999",  # neutral gray
]

old_palette_5 = ["#2384B8", "#B51D9E", "#3D9950", "#D6A44D", "#DB4338"]
old_palette_8 = ["#2384B8", "#B51D9E", "#3D9950", "#D6A44D", "#DB4338", "#504195", "#5AA137", "#44ADA8"]
old_palette_13 = old_palette_8 + [
    "#A98D9A", "#556F7D", "#5CAE4D", "#AD6D45", "#C9C985"
]

def set_plot_style():
    style_rc = {
        #"axes.facecolor": "#F5F5F9",                 # color of plotting area
        #"axes.edgecolor": "white",                   # color of axes edges
        "axes.grid": True,
        #"axes.axisbelow": True,
        #"axes.labelcolor": ".15",
        #"figure.facecolor": "white",                 # color of figure background
        "grid.color": "#DEDEDE",                     # color of background grid
        #"grid.linestyle": "-",                       # linestyle of background grid
        #"text.color": ".15",
        #"xtick.color": ".15",
        #"ytick.color": ".15",
        #"xtick.direction": "out",                    # location of ticks
        #"ytick.direction": "out",                    # location of ticks
        #"lines.solid_capstyle": "round",
        #"patch.edgecolor": "w",
        #"patch.force_edgecolor": True,
        #"image.cmap": "rocket",
        "font.family": ["sans-serif"],               # what font family to use
        "font.sans-serif": ['Arial',
                            'DejaVu Sans',           # what fonts should be used
                            'Liberation Sans',
                            'Bitstream Vera Sans',
                            'sans-serif'
                           ],
        "xtick.bottom": True,                        # whether ticks to be displayed
        #"xtick.top": False,                          # whether ticks to be displayed
        "ytick.left": True,                          # whether ticks to be displayed
        #"ytick.right": False,                        # whether ticks to be displayed
        #'axes.spines.left': True,                    # lines of figure casing
        #'axes.spines.bottom': True,                  # lines of figure casing
        #'axes.spines.right': True,                   # lines of figure casing
        #'axes.spines.top': True                      # lines of figure casing
        }
    context_rc = {
        #"font.size": 12.0,
        "axes.lablesize": 10.0,                      # thickness of axes labels
        #"axes.titlesize": 35.0,
        "xtick.labelsize": 20.0,                     # size of tick labels
        "ytick.labelsize": 20.0,                     # size of tick labels
        "legend.fontsize": 20.0,                     # size of legend labels
        "axes.linewidth": 1.25,                      # thickness of figure casing
        #"grid.linewidth": 1.0,                       # thickness of background grid
        "lines.linewidth": 2.0,                      # thickness of lines in plot
        "xtick.major.width": 2.0,                    # thickness of ticks
        "ytick.major.width": 2.0,                    # thickness of ticks
        #"xtick.minor.width": 1.0,
        #"ytick.minor.width": 1.0,
        "xtick.major.size": 12.0,                    # length of ticks
        "ytick.major.size": 12.0,                    # length of ticks
        #"xtick.minor.size": 4.0,
        #"ytick.minor.size": 4.0,
        #"legend.title_fontsize": 12.0
        }
    f_scaling = 2.0                                  # further size adjustment
    
    sns.set_style(
        "white",
        style_rc)
    sns.set_context(
        "notebook",
        font_scale=f_scaling,
        rc=context_rc)

class SeabornFig2Grid():

    def __init__(self, seaborngrid, fig,  subplot_spec):
        self.fig = fig
        self.sg = seaborngrid
        self.subplot = subplot_spec
        if isinstance(self.sg, sns.axisgrid.FacetGrid) or \
            isinstance(self.sg, sns.axisgrid.PairGrid):
            self._movegrid()
        elif isinstance(self.sg, sns.axisgrid.JointGrid):
            self._movejointgrid()
        self._finalize()

    def _movegrid(self):
        """ Move PairGrid or Facetgrid """
        self._resize()
        n = self.sg.axes.shape[0]
        m = self.sg.axes.shape[1]
        self.subgrid = gridspec.GridSpecFromSubplotSpec(n,m, subplot_spec=self.subplot)
        for i in range(n):
            for j in range(m):
                self._moveaxes(self.sg.axes[i,j], self.subgrid[i,j])

    def _movejointgrid(self):
        """ Move Jointgrid """
        h= self.sg.ax_joint.get_position().height
        h2= self.sg.ax_marg_x.get_position().height
        r = int(np.round(h/h2))
        self._resize()
        self.subgrid = gridspec.GridSpecFromSubplotSpec(r+1,r+1, subplot_spec=self.subplot)

        self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
        self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
        self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])

    def _moveaxes(self, ax, gs):
        #https://stackoverflow.com/a/46906599/4124317
        ax.remove()
        ax.figure=self.fig
        self.fig.axes.append(ax)
        self.fig.add_axes(ax)
        ax._subplotspec = gs
        ax.set_position(gs.get_position(self.fig))
        ax.set_subplotspec(gs)

    def _finalize(self):
        plt.close(self.sg.fig)
        self.fig.canvas.mpl_connect("resize_event", self._resize)
        self.fig.canvas.draw()

    def _resize(self, evt=None):
        self.sg.fig.set_size_inches(self.fig.get_size_inches())