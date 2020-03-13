import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time

matplotlib.rcParams['font.sans-serif'] = "Helvetica"
matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.rcParams['font.size'] = 17
matplotlib.rcParams['figure.figsize'] = (8.2, 8)

def DrawLineChart(data, fname_out_pdf, title='Title', legends=None, xlabel='Threshold', ylabel='Value'):
    # data is of size M x N, where M is number of methods, N is the values, evenly distributed from [0,1]
    assert data.shape[0] == len(legends), 'data.shape[0] == len(legends)'
    M, N = data.shape
    x = np.linspace(0, 1, N)

    data = data[:,1:-1]
    x = x[1:-1]

    fig = plt.figure()
    fig.show()
    plt.grid(linestyle='dotted')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    ax = fig.gca()
    rx = x.max() - x.min()
    ax.set_xlim(x.min() - rx * 0.1, x.max() + rx * 0.1)
    ry = data.max() - data.min()
    ax.set_ylim(data.min() - ry * 0.3, data.max() + ry * 0.1)

    colors = [
        [31, 119, 180, 220],
        [119, 31, 180, 220],
        [242, 48, 97, 220],
        [180, 119, 31, 220],
        [95, 138, 12, 220],
        [17, 87, 243, 220],
        [139, 18, 129, 220],
        [31, 119, 180, 220],
        [119, 31, 180, 220],
        [180, 119, 31, 255],
        [95, 138, 12, 255],
        [139, 18, 129, 255],
        [242, 48, 97, 255],
        [17, 87, 243, 255],
    ]

    markers = ['o', 'D', '^']

    for k in range(M):
        marker = markers[k % 3]
        color = [x/255.0 for x in colors[k % 7]]
        color_fill = [*color[0:3], color[3] * 0.5]
        ax.plot([], [], marker, linestyle='--', color=color, lw=1.7, mew=1.6, ms=8.5, mec=color, mfc=color_fill, label=legends[k])
        line = ax.lines[k]
        line.set_data(x, data[k])

    ax.legend(loc='best', shadow=False, fontsize='small', fancybox=True, framealpha=0.75)
    ax.xaxis.set_label_coords(0.5, 0.05)
    ax.yaxis.set_label_coords(0.05, 0.5)

    fig.canvas.draw()
    fig.savefig(fname_out_pdf)