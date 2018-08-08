import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcdefaults()
plt.style.use('C:\\JAModel\\ja_model\\ja_model.mplstyle')

def x_label_setter(label, axes):
    axes.set_xlabel(label, fontweight="bold")

def y_label_setter(label, axes):
    axes.set_ylabel(label, fontweight="bold")

def simpleaxis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
