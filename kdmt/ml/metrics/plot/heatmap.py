from kdmt.color import truncate_colormap

def default_heatmap():
    import matplotlib.pyplot as plt
    return truncate_colormap(plt.cm.OrRd, 0.1, 0.7)

