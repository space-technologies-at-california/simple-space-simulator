import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.ticker import MultipleLocator
import numpy as np
import math
from abc import ABC, abstractmethod

import simple_space_simulator.utils as utils


class SimPlot(ABC):
    """
    Parent class for all simple space simulator plot subclasses
    """

    def __init__(self, is_3d=False):
        self.is_3d = is_3d

    @abstractmethod
    def build(self, states, time_stamps, ax):
        pass


class CartesianPlot(SimPlot):
    """
    This plot plots the position in x, y, z in meters in ECEF along time in seconds
    """

    def build(self, states, time_stamps, ax):
        x = [state.get_x() for state in states]
        y = [state.get_y() for state in states]
        z = [state.get_z() for state in states]
        ax.plot(time_stamps, x, color="red", label='x')
        ax.plot(time_stamps, y, color="green", label='y')
        ax.plot(time_stamps, z, color="blue", label='z')
        ax.set_title("Cartesian Coordinates Plot")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("position (m)")

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels)


class CartesianVelocityPlot(SimPlot):
    """
    This plot plots the velocity in the x, y, z in m/s in ECEF along time in seconds
    """

    def build(self, states, time_stamps, ax):
        dx = [state.get_dx() for state in states]
        dy = [state.get_dy() for state in states]
        dz = [state.get_dz() for state in states]
        ax.plot(time_stamps, dx, color="red", label='dx')
        ax.plot(time_stamps, dy, color="green", label='dy')
        ax.plot(time_stamps, dz, color="blue", label='dz')
        ax.set_title("Cartesian Velocity Plot")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("velocity (m/s)")

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels)


class SphericalPlot(SimPlot):
    """
    This plot plots r, lat, lon in meters and radians along time in seconds
    """

    def build(self, states, time_stamps, ax):
        r = [state.get_r() for state in states]
        lon = [state.get_lon() for state in states]
        lat = [state.get_lat() for state in states]
        lns1 = ax.plot(time_stamps, lat, color="green", label='lat')
        lns2 = ax.plot(time_stamps, lon, color="blue", label='lon')
        ax.set_title("Spherical Coordinates Plot")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("angle (radians)")
        ax.yaxis.set_major_locator(MultipleLocator(base=np.pi / 2))

        ax2 = ax.twinx()
        lns3 = ax2.plot(time_stamps, r, color="red", label='radius')
        # this gives a better sense of scale for r by removing very small values
        ax2.set_ylim(ax2.get_ylim()[0] - 1, ax2.get_ylim()[1] + 1)

        lns = lns1 + lns2 + lns3
        labels = [ln.get_label() for ln in lns]
        ax.legend(lns, labels, loc='upper center')


class SphericalVelocityPlot(SimPlot):
    """
    This plot plots dr, dlat, dlon in m/s and rad/s along time in seconds
    """

    def build(self, states, time_stamps, ax):
        r = [state.get_dr() for state in states]
        lon = [state.get_dtheta() for state in states]
        lat = [state.get_dphi() for state in states]

        lns1 = ax.plot(time_stamps, lat, color="green", label='dphi')
        lns2 = ax.plot(time_stamps, lon, color="blue", label='dtheta')
        ax.set_title("Spherical Velocity Plot")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("angle (radians) / s")
        ax.yaxis.set_major_locator(MultipleLocator(base=np.pi / 2))

        ax2 = ax.twinx()
        lns3 = ax2.plot(time_stamps, r, color="red", label='dr')
        # this gives a better sense of scale for dr by removing very small values
        ax2.set_ylim(ax2.get_ylim()[0] - 0.1, ax2.get_ylim()[1] + 0.1)

        lns = lns1 + lns2 + lns3
        labels = [ln.get_label() for ln in lns]
        ax.legend(lns, labels)


class OrbitalPlot3D(SimPlot):
    """
    This plot plots the orbit of the satellite in 3D
    """

    def __init__(self, planet, show_planet=False, show_magnetic_field=False):
        super().__init__(is_3d=True)
        self.planet = planet
        self.show_planet = show_planet
        self.show_magnetic_field = show_magnetic_field

    def build(self, states, time_stamps, ax):
        x = [state.get_x() for state in states]
        y = [state.get_y() for state in states]
        z = [state.get_z() for state in states]
        ax.plot3D(x, y, z, 'red')

        if self.show_planet:
            count = 15  # keep 180 points along theta and phi
            # define a grid matching the map size, subsample along with pixels
            theta = np.linspace(0, np.pi, count)
            phi = np.linspace(0, 2 * np.pi, count)
            theta, phi = np.meshgrid(theta, phi)
            R = self.planet.radius
            # sphere
            x = R * np.sin(theta) * np.cos(phi)
            y = R * np.sin(theta) * np.sin(phi)
            z = R * np.cos(theta)
            ax.plot_surface(x.T, y.T, z.T, cstride=1, rstride=1)  # we've already pruned ourselves

        if self.show_magnetic_field:
            for i in range(0, len(states), len(states) // 10):
                state = states[i]
                magnetic_field_vector = self.planet.get_magnetic_field(state)
                magnetic_field_vector_norm = magnetic_field_vector / np.linalg.norm(magnetic_field_vector)
                magnetic_field_vector_norm *= 4e6  # this makes the vectors visible
                ax.quiver(state.get_x(), state.get_y(), state.get_z(), magnetic_field_vector_norm[0],
                          magnetic_field_vector_norm[1], magnetic_field_vector_norm[2], color='green')

        ax.set_title("Orbital Plot 3D")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        ax.set_xlim(-1.5e7, 1.5e7)
        ax.set_ylim(-1.5e7, 1.5e7)
        ax.set_zlim(-1.5e7, 1.5e7)


class OrbitalPlot2D(SimPlot):
    """
    This plot plots the orbit of the satellite in a 2D plane given by the inclination (default 0)
    """

    def __init__(self, planet, inclination=0):
        super().__init__()
        self.planet = planet
        self.inclination = inclination

    def build(self, states, time_stamps, ax):
        ax.add_artist(plt.Circle((0, 0), self.planet.radius, color='blue'))

        x = [state.get_x() for state in states]
        y = [state.get_y() for state in states]
        z = [state.get_z() for state in states]

        T = np.array([[1, 0, 0],
                      [0, math.cos(-self.inclination), -math.sin(-self.inclination)],
                      [0, math.sin(-self.inclination), math.cos(-self.inclination)]])
        for i in range(len(x)):
            p = np.dot(T, np.array([x[i], y[i], z[i]]))
            x[i] = p[0]
            y[i] = p[1]
            z[i] = p[2]

        ax.plot(x, y, color="red", label='x')
        ax.set_title("Orbital Plot 2D")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(-1.5e7, 1.5e7)
        ax.set_ylim(-1.5e7, 1.5e7)


class OrientationPlot(SimPlot):
    """
    This plot plots the orientation of the satellite itself
    """

    def build(self, states, time_stamps, ax):
        X, Y, Z = [], [], []
        for state in states:
            x, y, z = utils.quaternion_to_euler_angle(*state.get_orientation_quaternion())
            X.append(x)
            Y.append(y)
            Z.append(z)

        ax.plot(time_stamps, X, color="red", label='roll')
        ax.plot(time_stamps, Y, color="green", label='pitch')
        ax.plot(time_stamps, Z, color="blue", label='yaw')

        ax.set_title("Orientation Plot")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("radians")

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels)


class AngularVelocityPlot(SimPlot):
    """
    This plot plots the angular velocity of the satellite and the angular momentum
    """
    def __init__(self, cubesat):
        super().__init__()
        self.cubesat = cubesat

    def build(self, states, time_stamps, ax):
        X, Y, Z = [], [], []
        angular_momentum = []
        for state in states:
            w = state.get_angular_velocity_vector()
            X.append(w[0])
            Y.append(w[1])
            Z.append(w[2])
            angular_momentum.append(np.sum(np.dot(self.cubesat.inertia, w)))

        ax.set_title("Angular Velocity Plot")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("radians / s")

        lns1 = ax.plot(time_stamps, X, color="red", label='v roll')
        lns2 = ax.plot(time_stamps, Y, color="green", label='v pitch')
        lns3 = ax.plot(time_stamps, Z, color="blue", label='v yaw')

        ax2 = ax.twinx()
        lns4 = ax2.plot(time_stamps, angular_momentum, color="orange", label='angular momentum')

        lns = lns1 + lns2 + lns3 + lns4
        labels = [ln.get_label() for ln in lns]
        ax.legend(lns, labels)


class MagneticFieldPlot(SimPlot):
    """
    This plot plots the strength of the x, y, and z components of the magnetic field in tesla
    """

    def __init__(self, planet):
        super().__init__()
        self.planet = planet

    def build(self, states, time_stamps, ax):
        x, y, z = [], [], []
        for state in states:
            field = self.planet.get_magnetic_field(state)
            x.append(field[0])
            y.append(field[1])
            z.append(field[2])
        ax.plot(time_stamps, x, color="red", label='x')
        ax.plot(time_stamps, y, color="green", label='y')
        ax.plot(time_stamps, z, color="blue", label='z')
        ax.set_title("Magnetic Field Plot")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("field strength (T)")

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels)


"""
Animated plots
"""


class AnimatedPlot(SimPlot, ABC):

    @abstractmethod
    def update(self):
        pass


class OrientationPlotAnimated(AnimatedPlot):
    def __init__(self, cubesat, draw_axes=True, draw_magnetic_dipoles=True):
        super().__init__(is_3d=True)
        self.cubesat = cubesat
        self.draw_axes = draw_axes
        self.draw_magnetic_dipoles = draw_magnetic_dipoles
        self.collection = Poly3DCollection(
            utils.points_to_verts(self.cubesat.points), facecolors='grey',
            linewidths=1, edgecolors='black', alpha=0.6)

    def build(self, states, time_stamps, ax):
        self.states = states
        self.time_stamps = time_stamps
        self.ax = ax

        self.axes_size = np.max(self.cubesat.points) * 2 * 2
        self.ax.set_xlim3d([-self.axes_size, self.axes_size])
        self.ax.set_xlabel('X')
        self.ax.set_ylim3d([-self.axes_size, self.axes_size])
        self.ax.set_ylabel('Y')
        self.ax.set_zlim3d([-self.axes_size, self.axes_size])
        self.ax.set_zlabel('Z')
        self.ax.add_collection3d(self.collection)

        self.x = np.array([self.axes_size / 2, 0, 0])
        self.y = np.array([0, self.axes_size / 2, 0])
        self.z = np.array([0, 0, self.axes_size / 2])

        # static reference axes
        self.ax.quiver(0, 0, 0, self.x[0], self.x[1], self.x[2], color='red')
        self.ax.quiver(0, 0, 0, self.y[0], self.y[1], self.y[2], color='green')
        self.ax.quiver(0, 0, 0, self.z[0], self.z[1], self.z[2], color='blue')

        # dynamic local axes
        if self.draw_axes:
            self.x_quiver = self.ax.quiver(0, 0, 0, self.axes_size / 2, 0, 10, color='red')
            self.y_quiver = self.ax.quiver(0, 0, 0, 0, self.axes_size / 2, 0, color='green')
            self.z_quiver = self.ax.quiver(0, 0, 0, 0, 0, self.axes_size / 2, color='blue')

        if self.draw_magnetic_dipoles:
            self.cubesat_magnetic_dipole = self.ax.quiver(0, 0, 0, 0, 0, 0, color='purple')

    def update_axes(self, state):
        self.x_quiver.remove()
        self.y_quiver.remove()
        self.z_quiver.remove()
        x = utils.quaternion_rotate(state.get_orientation_quaternion(), self.x)
        y = utils.quaternion_rotate(state.get_orientation_quaternion(), self.y)
        z = utils.quaternion_rotate(state.get_orientation_quaternion(), self.z)
        self.x_quiver = self.ax.quiver(0, 0, 0, x[0], x[1], x[2], color='red')
        self.y_quiver = self.ax.quiver(0, 0, 0, y[0], y[1], y[2], color='green')
        self.z_quiver = self.ax.quiver(0, 0, 0, z[0], z[1], z[2], color='blue')

    def update_magnetic_dipoles(self, state):
        self.cubesat_magnetic_dipole.remove()
        cubesat_dipole = self.cubesat.get_magnetic_dipole(state)
        self.cubesat_magnetic_dipole = self.ax.quiver(0, 0, 0, cubesat_dipole[0], cubesat_dipole[1], cubesat_dipole[2],
                                                      color='purple')

    def update(self):
        for state, time in zip(self.states, self.time_stamps):
            points = []
            for point in self.cubesat.points:

                if self.draw_axes:
                    self.update_axes(state)

                if self.draw_magnetic_dipoles:
                    self.update_magnetic_dipoles(state)

                points.append(utils.quaternion_rotate(state.get_orientation_quaternion(), point))
            self.ax.collections[0].set_verts(utils.points_to_verts(np.array(points)))
            yield state, time
