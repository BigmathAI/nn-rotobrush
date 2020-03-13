import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time

matplotlib.rcParams['font.sans-serif'] = "Helvetica"
matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.rcParams['figure.figsize'] = (18, 6)

class fig_handle:
    def __init__(self, title='title', xlabel='xlabel', ylabel='ylabel',
                 len=None, display=True, cache='./figure.cache', filename='./fig.pdf'):
        self.display = display
        if not display:
            plt.switch_backend('agg')

        self.fig = plt.figure()
        self.fig.show()
        self.len = len

        plt.grid(linestyle='dotted')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        ax = self.fig.gca()
        ax.plot([], [], 'o-', color=(0.4,0.5,1.0), lw=0.8, mew=0.8, ms=5, mfc='none')
        self.cache_file = self.restore(cache)
        self.filename_for_save = filename

    def __del__(self):
        self.cache_file.close()

    def restore(self, cache):
        try:
            f = open(cache, 'r+')
            lines = f.readlines()
            def _convert(line):
                x, y = line.rstrip().split(' ')
                return int(x), float(y)
            xs, ys = zip(*[_convert(line) for line in lines])
            self.fig.gca().lines[0].set_data(xs, ys)
        except:
            f = open(cache, 'w')
        return f

    def update(self, x, y):
        fig = self.fig
        ax = fig.gca()
        line = ax.lines[0]
        x_data = np.append(line.get_xdata(), x)
        y_data = np.append(line.get_ydata(), y)
        if self.len is not None:
            x_data = x_data[-self.len:]
            y_data = y_data[-self.len:]

        xr = x_data.max() - x_data.min() + 0.001
        yr = y_data.max() - y_data.min() + 0.001
        s = 0.05
        ax.set_xlim(x_data.min() - s * xr, x_data.max() + s * xr)
        ax.set_ylim(y_data.min() - s * yr, y_data.max() + s * yr)

        line.set_data(x_data, y_data)
        fig.canvas.draw()
        if self.display:
            fig.canvas.flush_events()
        time.sleep(0.05)
        self.cache_file.write('%d %f\n' % (x, y))

    def savefig(self, filename=None):
        if filename is None:
            self.fig.savefig(self.filename_for_save)
        else:
            self.fig.savefig(filename)

    def update_and_savefig(self, x, y, filename=None):
        self.update(x, y)
        self.savefig(filename)