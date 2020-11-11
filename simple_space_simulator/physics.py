import numpy as np
import pyIGRF

import simple_space_simulator.utils as utils
import simple_space_simulator.cubesat as cubesat
from simple_space_simulator import constants


class Simulator:
    """
    This class handles the step by step calculations required to simulate an orbiting body and
    the interactions with multiple forcers, torquers, and accelerators
    """

    def __init__(self, cubesat, planet, state, dt=0.1):
        """
        Parameters
        ----------
        cubesat : Cubesat
            The orbiting cubesat object
        planet : Planet
            The planet that the cubesat object will be orbiting
        state : State
            Initial state of the simulation
        dt : float, optional
            The time between steps of the simulation

        Returns
        -------
        Simulator
            A simulator object that can be stepped through and used with a renderer object
        """
        self.dt = dt
        self.cubesat = cubesat
        self.planet = planet
        self.elapsed_time = 0
        self.state = state
        # functions that take in a state and returns a force vector <Fx,Fy,Fz> acting on the COM of the cubesat
        self.forces = []
        # functions that take in a state and returns an acceleration vector <ax,ay,az> acting on the COM of the cubesat
        self.accelerations = []
        # function that takes in a state and returns a torque vector <tx, ty, tz> acting around the COM of the
        # cubesat with inertia I
        self.torques = []
        # function that takes in a state and returns an angular acceleration vector <alphax, alphay, alphaz> acting
        # around the COM
        self.angular_accelerations = []

        # The new state is provided by the following equation
        # xk = Axk-1 + Buk-1 + wk-1
        # u is the vector sum of all the accelerations

        # 3D state matrix - A
        self.A = np.array([[1, 0, 0, self.dt, 0, 0],
                           [0, 1, 0, 0, self.dt, 0],
                           [0, 0, 1, 0, 0, self.dt],
                           [0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 1]])

        # 3D control matrix - B
        self.B = np.array([[1 / 2 * self.dt ** 2, 0, 0],
                           [0, 1 / 2 * self.dt ** 2, 0],
                           [0, 0, 1 / 2 * self.dt ** 2],
                           [self.dt, 0, 0],
                           [0, self.dt, 0],
                           [0, 0, self.dt]])

    def step(self):
        """
        Computes the accelerations for the current time step given state S.
        Returns
        -------
        tuple
            A  tuple that contains the elapsed time and the new state
        """
        net_force = np.zeros(3)
        for forcer in self.forces:
            net_force += forcer(self.state, self.cubesat, self.planet)

        net_acc = np.zeros(3)
        for accelerator in self.accelerations:
            net_acc += accelerator(self.state, self.cubesat, self.planet)

        net_acc += net_force / self.cubesat.mass

        net_torque = np.zeros(3)
        for torquer in self.torques:
            net_torque += torquer(self.state, self.cubesat, self.planet)

        net_angular_acc = np.zeros(3)
        for angular_accelerator in self.angular_accelerations:
            net_angular_acc += angular_accelerator(self.state, self.cubesat, self.planet)

        net_angular_acc += np.dot(self.cubesat.inertia_inv, net_torque)

        angular_velocity = self.state.get_angular_velocity_vector() + net_angular_acc * self.dt

        # https://math.stackexchange.com/questions/39553/how-do-i-apply-an-angular-velocity-vector3-to-a-unit-quaternion-orientation
        w = angular_velocity * self.dt
        a = 1/2 * net_angular_acc * self.dt ** 2
        w_norm = np.linalg.norm(w)
        w_f = w * np.sin(w_norm / 2) / w_norm if w_norm != 0 else [0, 0, 0]
        q = utils.quaternion_multiply(np.array([np.cos(w_norm / 2), w_f[0], w_f[1], w_f[2]]),
                                      self.state.get_orientation_quaternion())
        # q /= np.linalg.norm(q)

        # A*state + B*acceleration
        self.state = cubesat.state_from_vectors(
            np.dot(self.A, self.state.get_cartesian_state_vector()) + np.dot(self.B, net_acc), q, angular_velocity)
        self.elapsed_time += self.dt
        return self.elapsed_time, self.state

    def add_forcer(self, forcer):
        self.forces.append(forcer)

    def add_accelerator(self, accelerator):
        self.accelerations.append(accelerator)

    def add_torquer(self, torquer):
        self.torques.append(torquer)

    def add_angular_accelerator(self, angular_accelerator):
        self.angular_accelerations.append(angular_accelerator)


class Planet:
    def __init__(self, mass, radius, magnetic_field_model=pyIGRF):
        self.radius = radius  # m
        self.mass = mass  # kg
        self.magnetic_field_model = magnetic_field_model

    def get_gravitational_acceleration(self, state):
        r_hat = state.state_vector[:3] / np.linalg.norm(state.state_vector[:3])
        a = -constants.G * self.mass / np.linalg.norm(state.state_vector[:3]) ** 2 * r_hat
        return a

    def get_magnetic_field(self, state, ecef=True):
        # D: declination (+ve east)
        # I: inclination (+ve down)
        # H: horizontal intensity
        # X: north component
        # Y: east component
        # Z: vertical component (+ve down)
        # F: total intensity
        # unit: degree or nT
        s = self.magnetic_field_model.igrf_value(
            state.get_lat(), state.get_lon(), state.get_r() - constants.R_EARTH, 1999)
        magnetic_field_vector = np.array([s[3], s[4], s[5]]) / 1e9  # converted to T
        if ecef:
            return state.ned_to_ecef(magnetic_field_vector)
        return state
