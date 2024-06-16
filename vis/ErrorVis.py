import numpy as np
from matplotlib.lines import Line2D

from vis.ScatterVisualizer import ScatterVisualizer


class ErrorVis(ScatterVisualizer):
    def __init__(
        self,
        interactive=True,
        xlim=0,
        ylim=0,
        log_scale=True,
        offset=50,
        line_combine=True,
        sexify=False,
    ):
        """

        :param interactive:
        :param xlim:
        :param ylim:
        :param log_scale:
        :param offset:
        :param line_combine:
        """
        super().__init__(
            interactive=interactive,
            xlim=xlim,
            ylim=ylim,
            offset=offset,
            log_scale=log_scale,
            sexify=sexify,
        )
        if interactive:
            self.set_labels(xlabel="(iteration)", ylabel="log(error)")
        self.line_combine = line_combine
        self.offset = offset

    def update_with_point(self, x=0, y=0):
        """
        -> also draws line if set in initializer!
        :param p:
        :return:
        """
        p = np.zeros(2)
        p[0], p[1] = x, y
        er_array = self.my_plot.get_offsets()
        xvals = er_array[::2]
        yvals = er_array[1::2]

        ## did work but not anymore ;(
        # if len(yvals) and self.line_combine:
        #     self.ax.add_line(Line2D([er_array[-1][0], x],
        #                             [er_array[-1][1], y],
        #                             linestyle='dotted',
        #                             color='blue', linewidth=0.7))
        if len(yvals) and self.line_combine:
            self.ax.add_line(
                Line2D(
                    [xvals[-1][0], x],
                    [yvals[-1][1], y],
                    linestyle="dotted",
                    color="blue",
                    linewidth=0.7,
                )
            )

        er_array = np.append(er_array, p)
        er_array = np.c_[er_array[::2], er_array[1::2]]

        self.my_plot.set_offsets(er_array)
        self.my_plot.set_sizes([30.5] * len(er_array))

        self.ax.set_ylim(
            [
                np.min(er_array, axis=0)[1] - self.offset,
                np.max(er_array, axis=0)[1] + self.offset,
            ]
        )
        self.fig.canvas.draw()
