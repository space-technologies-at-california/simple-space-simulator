import numpy as np
import math

from simple_space_simulator import constants

"""
Orbital equations helper functions 
"""


def circular_orbit_velocity(altitude, mass=constants.M_EARTH, radius=constants.R_EARTH):
    """
    Returns the velocity required to maintain a circular orbit at a given altitude

    Parameters
    ----------
    altitude : float
        The distance above the ground that the object is positioned
    mass : float, optional
        Mass of the body being orbited (default is the mass of earth)
    radius : float, optional
        Radius of the body being orbited (default is the great sphere radius of earth)

    Returns
    -------
    float
        The velocity required for a perfectly circular orbit at a specified altitude
    """
    return math.sqrt(constants.G * mass / (altitude + radius))


def inclination_to_cartesian_velocity(speed, inclination):
    """
    Returns the components of the velocity vector required orbit at a specific inclination at a given speed

    Parameters
    ----------
    speed : float
        The positive speed in m/s at which the satellite will be orbiting
    inclination : float
        The angle of inclination in radians from horizontal that the satellite will orbit along

    Returns
    -------
    tuple
        (vx, vy, vz)
    """
    return 0, abs(speed) * np.cos(inclination), abs(speed) * np.sin(inclination)

"""
Quaternion helper functions  
"""


def quaternion_multiply(quaternion1, quaternion0):
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)


def quaternion_to_euler_angle(w, x, y, z):
    ysqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = np.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)

    t2 = np.clip(t2, a_min=-1.0, a_max=1.0)
    Y = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = np.arctan2(t3, t4)

    return X, Y, Z


def euler_to_quaternion(roll, pitch, yaw):
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)

    return qw, qx, qy, qz
