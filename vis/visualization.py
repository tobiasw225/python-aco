import time

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def animate(x_values, y_values):
    fig, ax = plt.subplots()
    scat = ax.scatter(x_values[0], y_values[0], c="b", s=5)

    def update(frame: int):
        ax.add_line(Line2D(x_values[frame], y_values[frame]))
        fig.canvas.draw()
        while True:
            try:
                ax.lines[0].remove()
            except IndexError:
                break
        time.sleep(1)
        return scat

    _ = animation.FuncAnimation(fig=fig, func=update, frames=10, interval=10)

    plt.show()
