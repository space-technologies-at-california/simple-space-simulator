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
    assert isinstance(altitude, (int, float)) and altitude > 0, "altitude must be a positive value"
    assert isinstance(mass, (int, float)) and mass > 0, "mass must be a positive value"
    assert isinstance(radius, (int, float)) and radius > 0, "radius must be a positive value"
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
    assert isinstance(speed, (int, float)) and speed > 0, "speed must be a positive value"
    assert isinstance(inclination, (int, float)), "inclination must be a radian value"
    return 0, abs(speed) * np.cos(inclination), abs(speed) * np.sin(inclination)


def steps_per_orbit(altitude, time_per_step, radius=constants.R_EARTH):
    """
    Returns the number of steps for a complete orbit based on constant speed at constant altitude model

    Parameters
    ----------
    altitude : float
        The altitude in meters above the center of the body being orbited.
    time_per_step : float
        The amount of time in seconds of each step
    radius : float, optional
        The radius of the body being orbited

    Returns
    -------
    int
        number of steps
    """
    assert isinstance(altitude, (int, float)) and altitude > 0, "altitude must be a positive value"
    assert isinstance(time_per_step, (int, float)) and time_per_step > 0, "time_per_step must be a positive value"
    assert isinstance(radius, (int, float)) and radius > 0, "radius must be a positive value"
    return int(2 * math.pi * math.sqrt((radius + altitude) ** 3 / (constants.G * constants.M_EARTH)) / time_per_step)


"""
Quaternion helper functions
"""


def quaternion_multiply(quaternion1, quaternion0):
    assert isinstance(quaternion1, np.ndarray) and quaternion1.ndim == 1, "quaternion1 must be a vector"
    assert isinstance(quaternion0, np.ndarray) and quaternion0.ndim == 1, "quaternion0 must be a vector"
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)


def quaternion_to_euler_angle(w, x, y, z):
    assert isinstance(w, (int, float)) and isinstance(x, (int, float)) and isinstance(y, (int,
        float)) and isinstance(z, (int, float)), "w x y z must be int or float values"
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
    assert isinstance(roll, (int, float)) and isinstance(pitch, (int, float)) and isinstance(yaw, (int, float)), \
        "roll pitch yaw must be int or float values"
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)

    return qw, qx, qy, qz


def quaternion_conjugate(q):
    assert isinstance(q, np.ndarray) and q.ndim == 1, "q must be a vector"
    return np.array([q[0], -q[1], -q[2], -q[3]])


# rotate v (x,y,z) by q (w,x,y,z)
def quaternion_rotate(q, v):
    assert isinstance(q, np.ndarray) and q.ndim == 1, "q must be a vector"
    assert isinstance(v, np.ndarray) and v.ndim == 1, "v must be a vector"
    return quaternion_multiply(quaternion_multiply(q, np.append([0], v)), quaternion_conjugate(q))[1:]


# Function for computing the vertices used in rendering in matplotlib
def points_to_verts(points):
    assert isinstance(points, np.ndarray), "points must be an array"
    return [[points[0], points[1], points[2], points[3]],
            [points[4], points[5], points[6], points[7]],
            [points[0], points[1], points[5], points[4]],
            [points[2], points[3], points[7], points[6]],
            [points[1], points[2], points[6], points[5]],
            [points[4], points[7], points[3], points[0]]]
