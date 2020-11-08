import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import math
from abc import ABC, abstractmethod

import simple_space_simulator.utils as utils
import simple_space_simulator.constants as constants
import pyIGRF


class SimPlot(ABC):
    """
    Parent class for all simple space simulator plot subclasses
    """

    def __init__(self):
        self.is_3d = False

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

        lns = lns1 + lns2 + lns3
        labels = [ln.get_label() for ln in lns]
        ax.legend(lns, labels)


class OrbitalPlot3D(SimPlot):
    """
    This plot plots the orbit of the satellite in 3D
    """

    def __init__(self, planet, show_planet=False, show_magnetic_field=False):
        self.is_3d = True
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
            for i in range(0, len(states), len(states)//10):
                state = states[i]
                s = pyIGRF.igrf_value(state.get_lat(), state.get_lon(), state.get_r() - constants.R_EARTH, 1999)
                magnetic_field_vector = np.array([s[3], s[4], s[5]])
                magnetic_field_vector_norm = magnetic_field_vector / np.linalg.norm(magnetic_field_vector)
                magnetic_field_vector_norm *= 4000000  # size of vector is relative
                magnetic_field_vector_norm = state.ned_to_ecef(magnetic_field_vector_norm)
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
            q = state.get_orientation_quaternion()
            x, y, z = utils.quaternion_to_euler_angle(q[0], q[1], q[2], q[3])
            X.append(x)
            Y.append(y)
            Z.append(z)

        ax.plot(time_stamps, X, color="red", label='x')
        ax.plot(time_stamps, Y, color="green", label='y')
        ax.plot(time_stamps, Z, color="blue", label='z')

        ax.set_title("Orientation Plot")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("radians")

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels)
