import numpy as np
import math
import scipy.spatial.transform as transform
import simple_space_simulator.utils as utils
import simple_space_simulator.physics as physics


class State:
    def __init__(self, x, y, z, dx, dy, dz, qw, qx, qy, qz, dqw, dqx, dqy, dqz):
        """
        Structure of the state vector (units are m, m/s, radians, radians/s)
        x, y, z, dx, dy, dz, roll, pitch, yaw, droll, dpitch, dyaw
        """
        assert isinstance(x, (int, float)) and isinstance(y, (int, float)) and isinstance(z, (int, float)), \
            "x y z must be int or float values"
        assert isinstance(dx, (int, float)) and isinstance(dy, (int, float)) and isinstance(dz, (int, float)), \
            "dx dy dz must be int or float values"
        assert isinstance(qw, (int, float)) and isinstance(qx, (int, float)) and \
               isinstance(qy, (int, float)) and isinstance(qz, (int, float)) and \
               np.linalg.norm(np.array([qw, qx, qy, qz])) - 1.0 < 0.1, \
               "orientation quaternion must be composed of int and floats and have norm near one"
        assert isinstance(qw, (int, float)) and isinstance(qx, (int, float)) and \
               isinstance(qy, (int, float)) and isinstance(qz, (int, float)), \
               "orientation derivative must contain integers or floats"

        self.state_vector = np.array([x, y, z, dx, dy, dz, qw, qx, qy, qz, dqw, dqx, dqy, dqz])

    def get_cartesian_state_vector(self):
        """
        Structure of the state vector (units are m and m/s)
        x, y, z, dx, dy, dz
        """
        return self.state_vector[:6]

    def get_ecef_position(self):
        return self.state_vector[:3]

    def get_orientation_euler(self):
        quat = self.get_orientation_quaternion()
        return transform.Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_euler('xyz')

    def get_orientation_quaternion(self):
        return self.state_vector[6:10] / np.linalg.norm(self.state_vector[6:10])

    def get_orientation_quaternion_conjugate(self):
        return utils.quaternion_conjugate(self.get_orientation_quaternion())

    def get_quaternion_derivative(self):
        return self.state_vector[10:14]

    def get_velocity_vector(self):
        return self.state_vector[3:6]

    def get_angular_velocity_vector(self):
        return utils.quaternion_multiply(self.get_orientation_quaternion_conjugate(),
                                         2*self.get_quaternion_derivative())[1:]

    def get_spherical_state_vector(self):
        """
        Structure of the earth referenced state vector (units are m, radians, and m/s)
        r (radius from the center of the plant), theta (latitude), phi (longitude), dr, dtheta, dphi

        Good reference http://dynref.engr.illinois.edu/rvs.html
        https://physics.stackexchange.com/questions/546479/conversion-of-cartesian-position-and-velocity-to-spherical-velocity
        """
        r = np.linalg.norm(self.get_ecef_position())
        theta = np.arctan2(self.get_y(), self.get_x())  # latitude
        phi = np.arccos(self.get_z() / r)  # longitude

        dr = (self.get_x() * self.get_dx() + self.get_y() * self.get_dy() + self.get_z() * self.get_dz()) / r
        dtheta = (self.get_x() * self.get_dy() - self.get_y() * self.get_dx()) / (self.get_x() ** 2 + self.get_y() ** 2)
        dphi = (self.get_z() * dr - r * self.get_dz()) / (r ** 2 * np.sqrt(1 - (self.get_z() / r) ** 2))

        return np.array([r, theta, phi, dr, dtheta, dphi])

    """    Getter methods for cartesian coordinates """

    def get_x(self):
        return self.state_vector[0]

    def get_y(self):
        return self.state_vector[1]

    def get_z(self):
        return self.state_vector[2]

    def get_dx(self):
        return self.state_vector[3]

    def get_dy(self):
        return self.state_vector[4]

    def get_dz(self):
        return self.state_vector[5]

    def get_roll(self):
        return self.get_orientation_euler()[0]

    def get_pitch(self):
        return self.get_orientation_euler()[1]

    def get_yaw(self):
        return self.get_orientation_euler()[2]

    def get_droll(self):
        return self.get_angular_velocity_vector()[0]

    def get_dpitch(self):
        return self.get_angular_velocity_vector()[1]

    def get_dyaw(self):
        return self.get_angular_velocity_vector()[2]

    # Getter methods for spherical coordinates
    def get_r(self):
        return self.get_spherical_state_vector()[0]

    def get_altitude(self, planet):
        assert isinstance(planet, physics.Planet), "planet must be a Planet object"
        return self.get_r() - planet.radius

    def get_theta(self):
        return self.get_spherical_state_vector()[1]

    def get_lat(self):
        return math.pi / 2 - self.get_phi()

    def get_phi(self):
        return self.get_spherical_state_vector()[2]

    def get_lon(self):
        return self.get_theta()

    def get_dr(self):
        return self.get_spherical_state_vector()[3]

    def get_dtheta(self):
        return self.get_spherical_state_vector()[4]

    def get_dphi(self):
        return self.get_spherical_state_vector()[5]

    # NED for magnetic field
    # np.dot(T, vector in NED) results in vector in ECEF
    def ned_to_ecef(self, vector):
        assert isinstance(vector, np.ndarray) and vector.ndim == 1, "vector must be a vector"
        lon = self.get_lon()
        lat = self.get_lat()
        DCMned = np.linalg.inv(np.array([[-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
                                         [-np.sin(lon), np.cos(lon), 0],
                                         [-np.cos(lat) * np.cos(lon), -np.cos(lat) * np.sin(lon), -np.sin(lat)]]))
        return np.dot(DCMned, vector)

    # np.dot(T, vector in ECEF) results in vector in NED
    def ecef_to_ned(self, vector):
        assert isinstance(vector, np.ndarray) and vector.ndim == 1, "vector must be a vector"
        lon = self.get_lon()
        lat = self.get_lat()
        DCMef = np.array([[-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
                          [-np.sin(lon), np.cos(lon), 0],
                          [-np.cos(lat) * np.cos(lon), -np.cos(lat) * np.sin(lon), -np.sin(lat)]])
        return np.dot(DCMef, vector)

    def __str__(self):
        return str(self.state_vector)


def state_from_vector(v):
    # v is [x, y, z, dx, dy, dz, r, p y, wr, wp, wy]
    return State(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11], v[12], v[13])


"""
TODO Sensor Deivces
"""


class SensorDevice:
    def __init__(self):
        pass

    def attach(self, cubesat, location):
        pass


"""
TODO Control Devices
"""


class ControlDevice:
    def __init__(self):
        pass

    def attach(self, cubesat):
        pass


class Cubesat:
    def __init__(self, mass, length=0.1, width=0.1, height=0.1, inertia=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])):
        """
        mass in kg
        length, width, height in m
        """
        assert isinstance(mass, (int, float)) and mass > 0, "mass must be a positive value"
        assert isinstance(length, (int, float)) and length > 0, "length must be a positive value"
        assert isinstance(width, (int, float)) and width > 0, "width must be a positive value"
        assert isinstance(height, (int, float)) and height > 0, "height must be a positive value"
        assert isinstance(inertia, np.ndarray) and inertia.ndim == 2, "inertia must be a 2d matrix"

        self.mass = mass
        self.inertia = inertia
        self.inertia_inv = np.linalg.inv(inertia)

        # static magnetic dipoles as vectors in 3d
        self.magnetic_dipoles = []

        # Dimensions
        self.length = length
        self.width = width
        self.height = height
        self.points = np.array([[-width / 2, -length / 2, -height / 2],
                                [width / 2, -length / 2, -height / 2],
                                [width / 2, length / 2, -height / 2],
                                [-width / 2, length / 2, -height / 2],
                                [-width / 2, -length / 2, height / 2],
                                [width / 2, -length / 2, height / 2],
                                [width / 2, length / 2, height / 2],
                                [-width / 2, length / 2, height / 2]])

    def add_magnetic_dipole(self, dipole):
        assert isinstance(dipole, np.ndarray) and dipole.ndim == 1, "dipole must be a vector"
        self.magnetic_dipoles.append(dipole)

    def get_magnetic_dipole(self, state):
        assert isinstance(state, State), "state must be a State object"
        dipole_sum = np.array([0.0, 0.0, 0.0])
        for dipole in self.magnetic_dipoles:
            dipole_sum += utils.quaternion_rotate(state.get_orientation_quaternion(), dipole)
        return dipole_sum
