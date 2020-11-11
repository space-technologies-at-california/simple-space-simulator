import numpy as np
import math
import simple_space_simulator.utils as utils


class State:
    def __init__(self, x, y, z, dx, dy, dz, qw, qx, qy, qz, wx, wy, wz):
        """
        Structure of the state vector (units are m and m/s)
        x, y, z, dx, dy, dz
        Rotation is expressed as theta as quaternion q (w, x, y, z)
        """
        # contains the state as a cartesian state vector
        self.state_vector = np.array([x, y, z, dx, dy, dz])
        self.w = np.array([wx, wy, wz])
        self.q = np.array([qw, qx, qy, qz])

    def get_cartesian_state_vector(self):
        """
        Structure of the state vector (units are m and m/s)
        x, y, z, dx, dy, dz
        """
        return self.state_vector

    def get_orientation_quaternion(self):
        return self.q

    def get_orientation_quaternion_conjugate(self):
        return np.array([self.q[0], -self.q[1], -self.q[2], -self.q[3]])

    def get_angular_velocity_vector(self):
        return self.w

    def get_spherical_state_vector(self):
        """
        Structure of the earth referenced state vector (units are m, radians, and m/s)
        r (radius from the center of the plant), theta (latitude), phi (longitude), dr, dtheta, dphi

        Good reference http://dynref.engr.illinois.edu/rvs.html
        https://physics.stackexchange.com/questions/546479/conversion-of-cartesian-position-and-velocity-to-spherical-velocity
        """
        r = np.linalg.norm(self.state_vector[:3])
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

    # Getter methods for spherical coordinates
    def get_r(self):
        return self.get_spherical_state_vector()[0]

    def get_altitude(self, planet):
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
        lon = self.get_lon()
        lat = self.get_lat()
        DCMned = np.linalg.inv(np.array([[-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
                                         [-np.sin(lon), np.cos(lon), 0],
                                         [-np.cos(lat) * np.cos(lon), -np.cos(lat) * np.sin(lon), -np.sin(lat)]]))
        return np.dot(DCMned, vector)

    # np.dot(T, vector in ECEF) results in vector in NED
    def ecef_to_ned(self, vector):
        lon = self.get_lon()
        lat = self.get_lat()
        DCMef = np.array([[-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
                          [-np.sin(lon), np.cos(lon), 0],
                          [-np.cos(lat) * np.cos(lon), -np.cos(lat) * np.sin(lon), -np.sin(lat)]])
        return np.dot(DCMef, vector)

    def get_roll_pitch_yaw(self):
        return utils.quaternion_to_euler_angle(self.q[0], self.q[1], self.q[2], self.q[3])


def state_from_vectors(state, q, w):
    return State(state[0], state[1], state[2], state[3], state[4], state[5], q[0], q[1], q[2], q[3], w[0], w[1], w[2])


"""
TODO Sensor Deivces
"""


class SensorDevice:
    def __init__(self):
        pass


"""
TODO Control Devices
"""


class ControlDevice:
    def __init__(self):
        pass


class Cubesat:
    def __init__(self, mass, length=0.1, width=0.1, height=0.1, inertia=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])):
        """
        mass in kg
        length, width, height in m
        """
        self.mass = mass
        self.inertia = inertia
        self.inertia_inv = np.linalg.inv(inertia)

        self.length = length
        self.width = width
        self.height = height
        self.points = np.array([[-width/2, -length/2, -height/2],
                                [width/2, -length/2, -height/2],
                                [width/2, length/2, -height/2],
                                [-width/2, length/2, -height/2],
                                [-width/2, -length/2, height/2],
                                [width/2, -length/2, height/2],
                                [width/2, length/2, height/2],
                                [-width/2, length/2, height/2]])
