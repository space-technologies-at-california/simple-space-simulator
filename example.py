import simple_space_simulator.physics as physics
import simple_space_simulator.cubesat as cubesat
import simple_space_simulator.renderer as renderer
import simple_space_simulator.plots as plots

import math
import numpy as np

def vcircular(altitude):
    """
    This is a test:
    >>> vcircular(100)
    7909.618746598862
    """
    return math.sqrt(physics.Consts.mu / (altitude + physics.Consts.R_earth))


class OrientationPlot:
    def __init__(self, inclination=0):
        self.is_3d = False

    def build(self, states, time_stamps, ax):
        X, Y, Z = [], [], []
        for state in states:
            q = state.get_orientation_quaternion()
            x, y, z = quaternion_to_euler_angle(q[0], q[1], q[2], q[3])
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


def gravitational_forcer(state, cubesat, planet):
    return planet.get_gravitational_acceleration(state)


def euler_to_quaternion(roll, pitch, yaw):
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)

    return (qw, qx, qy, qz)


def inclination_to_cartesian_velocity(speed, inclination):
    return (0, speed * math.cos(inclination), speed * math.sin(inclination))


qubesat = cubesat.Cubesat(1, length=0.1, width=0.1, height=0.1) # mass in kg, length, width, height in m
planet  = physics.Planet(physics.Consts.M_earth, physics.Consts.R_earth) # mass in kg, radius in meters

vx, vy, vz = inclination_to_cartesian_velocity(vcircular(physics.Consts.ISS_altitude), math.pi / 2)
qw, qx, qy, qz = euler_to_quaternion(0, 0, 0)
# x, y, z, dx, dy, dz, qw, qx, qy, qz, wx, wy, wz
state = cubesat.State(physics.Consts.ISS_altitude + physics.Consts.R_earth, 0, 0, vx, vy, vz, qw, qx, qy, qz, 0.001,
                      0.000, 0.000)
simulator = physics.Simulator(qubesat, planet, state, 1)
simulator.add_accelerator(gravitational_forcer)

stop_condition = lambda states, times: len(states) > 10000
r = renderer.Renderer()
r.run(simulator, stop_condition)
plot1 = plots.CartesianPlot()
r.add_plot(plot1)
plot2 = plots.CartesianVelocityPlot()
r.add_plot(plot2)
plot3 = plots.SphericalPlot()
r.add_plot(plot3)
print("Spherical Velocities are Incorrect")
plot4 = plots.SphericalVelocityPlot()
r.add_plot(plot4)
plot5 = plots.OrbitalPlot2D(planet, inclination=physics.Consts.ISS_inclination)
r.add_plot(plot5)
plot6 = plots.OrbitalPlot3D(planet)
r.add_plot(plot6)
plot7 = OrientationPlot()
r.add_plot(plot7)

r.render(figsize=(15, 25), columns=2)
