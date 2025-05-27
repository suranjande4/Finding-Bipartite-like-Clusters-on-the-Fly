# plot_style.py
import matplotlib.pyplot as plt
import seaborn as sns

def set_plot_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "legend.title_fontsize": 14,
        "lines.linewidth": 2.5,
        "lines.markersize": 8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "axes.linewidth": 1.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
