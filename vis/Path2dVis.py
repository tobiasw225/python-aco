import time

import numpy as np
from matplotlib.lines import Line2D

from vis.ScatterVisualizer import ScatterVisualizer


class Path2DVis(ScatterVisualizer):
    class __Path2DVis(ScatterVisualizer):
        """
        scatter-plot lines for shortes path-visualisation
        """

        def __init__(
            self,
            xymin=-50,
            xymax=1100,
            num_runs=1,
            offset=50,
            interactive=True,
            sleep_interval=0,
        ):
            """

            :param xymin:
            :param xymax: -> das is käse
            :param num_runs:
            :param offset:
            :param interactive:
            :param sleep_interval:
            """
            super().__init__(
                interactive=interactive,
                xlim=0,
                ylim=0,
                offset=offset,
                log_scale=False,
                sexify=False,
            )
            self.set_yx_lim([xymin, xymax], [xymin, xymax])
            self.num_runs = num_runs
            self.sleep_interval = sleep_interval
            self.my_plot.set_edgecolor("white")

        def set_point_size(self, point_size=12.5):
            """

            :param point_size:
            :return:
            """
            self.plot.set_sizes([point_size] * len(self.target_array))

        def plot_path(self, x=[], y=[]):
            """

            :param x:
            :param y:
            :return:
            """
            self.ax.add_line(Line2D(x, y))
            self.set_yx_lim(
                [np.min(x) - self.offset, np.max(x) + self.offset],
                [np.min(y) - self.offset, np.max(y) + self.offset],
            )
            # plt.savefig("/home/tobias/Bilder/tsp/path"+str(iteration)+".png")
            self.fig.canvas.draw()
            time.sleep(self.sleep_interval)
            while True:
                # bisschen unschoen.
                try:
                    self.ax.lines[0].remove()
                except IndexError:
                    break

    instance = None

    def __init__(
        self,
        xymin=-50,
        xymax=1100,
        num_runs=1,
        offset=50,
        interactive=True,
        sleep_interval=0,
    ):
        if not Path2DVis.instance:
            Path2DVis.instance = Path2DVis.__Path2DVis(
                xymin, xymax, num_runs, offset, interactive, sleep_interval
            )

    def __getattr__(self, name):
        return getattr(self.instance, name)
