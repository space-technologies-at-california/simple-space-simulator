import numpy as np
import pyIGRF
import scipy.integrate as integrate
import scipy.spatial.transform as transform

import simple_space_simulator.cubesat as cube
from simple_space_simulator import constants


class Simulator:
    """
    This class handles the step by step calculations required to simulate an orbiting body and
    the interactions with multiple forcers, torquers, and accelerators
    """

    def __init__(self, cubesat, planet, state, max_step=10):
        """
        Parameters
        ----------
        cubesat : Cubesat
            The orbiting cubesat object
        planet : Planet
            The planet that the cubesat object will be orbiting
        state : State
            Initial state of the simulation
        max_step : float, optional
            The time between steps of the simulation

        Returns
        -------
        Simulator
            A simulator object that can be stepped through and used with a renderer object
        """
        assert isinstance(cubesat, cube.Cubesat), "cubesat must be a Cubesat object"
        assert isinstance(planet, Planet), "planet must be a Planet object"
        assert isinstance(state, cube.State), "state must be a State object"
        assert isinstance(max_step, int) and max_step > 0, "max step must be a positive integer"
        self.max_step = max_step
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

    def compute_lateral_accelerations(self, state=None):
        if state is None:
            state = self.state

        net_force = np.zeros(3)
        for forcer in self.forces:
            net_force += forcer(state, self.cubesat, self.planet)

        net_acc = np.zeros(3)
        for accelerator in self.accelerations:
            net_acc += accelerator(state, self.cubesat, self.planet)

        net_acc += net_force / self.cubesat.mass

        return net_acc

    def compute_angular_accelerations(self, state=None):
        if state is None:
            state = self.state

        net_torque = np.zeros(3)
        for torquer in self.torques:
            net_torque += torquer(state, self.cubesat, self.planet)

        net_angular_acc = np.zeros(3)
        for angular_accelerator in self.angular_accelerations:
            net_angular_acc += angular_accelerator(state, self.cubesat, self.planet)

        net_angular_acc += np.dot(self.cubesat.inertia_inv, net_torque)

        return net_angular_acc

    def sample(self, t, y):
        """
        Takes in the time and state vector y and returns the derivative at that point
        to be used by an integrate such as scipy ivp_solver.

        State vector: (x, y, z, dx, dy, dz, roll, pitch, yaw, droll, dpitch, dyaw)
        """
        state = cube.state_from_vector(y)
        acceleration = self.compute_lateral_accelerations(state)
        angular_acceleration = self.compute_angular_accelerations(state)*0

        # https://www.euclideanspace.com/physics/kinematics/angularvelocity/
        R = transform.Rotation.from_euler('xyz', state.get_orientation_euler()).inv()
        # angular velocity in the ecef frame
        angular_velocity = R.apply(state.get_angular_velocity_vector())

        return (*state.get_velocity_vector(),
                *acceleration,
                *angular_velocity,
                *angular_acceleration)

    def step(self, start=0, stop=5000, sample_resolution=10):
        """
        Computes the states from the current time step at t start to the state a t stop using
         Runge-Kutta 45 (ode45) with scipy ivp.
        Returns
        -------
        tuple
            A tuple that contains the time steps of each state and the states themselves
        """

        # Good reference: https://www.marksmath.org/visualization/orbits/CentralOrbit.html
        sol = integrate.solve_ivp(
            self.sample,
            (start, stop),
            self.state.state_vector,
            t_eval=np.linspace(start, stop, (stop - start) * sample_resolution),
            max_step=self.max_step)

        # Convert state arrays into state objects for easier manipulation
        states = [cube.state_from_vector(y) for y in sol.y.T]
        return sol.t, states

    def add_forcer(self, forcer):
        assert callable(forcer), "forcer must be a function"
        self.forces.append(forcer)

    def add_accelerator(self, accelerator):
        assert callable(accelerator), "accelerator must be a function"
        self.accelerations.append(accelerator)

    def add_torquer(self, torquer):
        assert callable(torquer), "torquer must be a function"
        self.torques.append(torquer)

    def add_angular_accelerator(self, angular_accelerator):
        assert callable(angular_accelerator), "angular_accelerator must be a function"
        self.angular_accelerations.append(angular_accelerator)


class Planet:
    def __init__(self, mass, radius, magnetic_field_model=pyIGRF):
        assert isinstance(mass, (int, float)) and mass > 0, "mass must be a positive value"
        assert isinstance(radius, (int, float)) and radius > 0, "radius must be a positive value"
        self.radius = radius  # m
        self.mass = mass  # kg
        self.magnetic_field_model = magnetic_field_model

    def get_gravitational_acceleration(self, state):
        assert isinstance(state, cube.State), "state must be a State object"
        r_hat = state.state_vector[:3] / np.linalg.norm(state.state_vector[:3])
        a = -constants.G * self.mass / np.linalg.norm(state.state_vector[:3]) ** 2 * r_hat
        return a

    def get_magnetic_field(self, state, ecef=True):
        assert isinstance(state, cube.State), "state must be a State object"
        assert isinstance(ecef, bool), "ecef must be a boolean"
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
