import matplotlib.pyplot as plt
import simple_space_simulator.constants as constants
import simple_space_simulator.physics as physics
from simple_space_simulator.state import State
import time
import numpy as np


class Renderer:
    def __init__(self, resolution=10):
        self.resolution = resolution
        self.time_stamps = []
        self.states = []
        self.plots = []

    def run(self, simulator, stop_time, start_time=0):
        assert isinstance(simulator, physics.Simulator), "simulator must be a Simulator object"
        assert isinstance(stop_time, (int, float)) and stop_time > 0, "stop time must be a float or int > 0"
        self.time_stamps, self.states = simulator.run(start_time, stop_time, self.resolution)
        completion_msg = "Simulation Complete \n" \
                         "{0} seconds \n" \
                         "{1} minutes \n" \
                         "{2} days {3} hours {4} minutes {5} seconds"
        elapsed_time = self.time_stamps[-1]
        print(completion_msg.format(elapsed_time,
                                    round(elapsed_time / constants.SECONDS_PER_MINUTE),
                                    elapsed_time // constants.SECONDS_PER_DAY,
                                    elapsed_time % constants.SECONDS_PER_DAY // constants.SECONDS_PER_HOUR,
                                    elapsed_time % constants.SECONDS_PER_DAY %
                                    constants.SECONDS_PER_HOUR // constants.SECONDS_PER_MINUTE,
                                    elapsed_time % constants.SECONDS_PER_DAY %
                                    constants.SECONDS_PER_HOUR % constants.SECONDS_PER_MINUTE))

    def save(self, filename=f"{time.time()}.npy"):
        with open(filename, 'wb') as file:
            arr = np.array([self.time_stamps, [s.state_vector for s in self.states]], dtype=object)
            np.save(file, arr)

    def load(self, filename):
        with open(filename, 'rb') as file:
            arr = np.load(file, allow_pickle=True)
            self.time_stamps = arr[0]
            self.states = [State.state_from_vector(s) for s in arr[1]]

    def add_plot(self, plot):
        self.plots.append(plot)

    def clear_plots(self):
        self.plots = []

    def render(self, figsize=(7, 5), columns=3):
        assert isinstance(figsize, tuple) and isinstance(columns, int), \
            "invalid dimensions: figsize must be a tuple and columns must be an int"
        fig = plt.figure(figsize=figsize)
        rows = len(self.plots) // columns + (1 if len(self.plots) % columns > 0 else 0)
        for i, plot in enumerate(self.plots):
            if plot.is_3d:
                ax = fig.add_subplot(rows, columns, i + 1, projection='3d')
            else:
                ax = fig.add_subplot(rows, columns, i + 1)
            plot.build(self.states, self.time_stamps, ax)
        plt.show()

    def run_animated_plot(self, plot, speed=10, start_time=0, stop_time=float('inf'), figsize=(7, 5)):
        fig = plt.figure(figsize=figsize)
        if plot.is_3d:
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)
        step_size = (self.time_stamps[1] - self.time_stamps[0])
        plot.build(self.states, self.time_stamps, ax)
        last_time = time.time()
        print()
        for state, stamp in plot.update():
            if start_time < stamp:
                plt.pause(max(step_size / speed - (time.time() - last_time), 1e-6))

            now = time.time()
            print('\rRTF: %.2f' % (step_size / (now - last_time)), end='')
            last_time = now

            if stamp > stop_time:
                break
