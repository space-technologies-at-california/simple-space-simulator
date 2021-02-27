import numpy as np
import pyIGRF
import scipy.integrate as integrate
import simple_space_simulator.cubesat as cube
from simple_space_simulator.state import State
from simple_space_simulator import constants


class Simulator:
    """
    This class handles the step by step calculations required to simulate an orbiting body and
    the interactions with multiple forcers, torquers, and accelerators
    """

    def __init__(self, cubesat, planet, initial_state, max_step=10):
        """
        Parameters
        ----------
        cubesat : Cubesat
            The orbiting cubesat object
        planet : Planet
            The planet that the cubesat object will be orbiting
        initial_state : State
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
        assert isinstance(initial_state, State), "state must be a State object"
        assert isinstance(max_step, (int, float)) and max_step > 0, "max step must be a positive float or integer"

        self.planet = planet
        self.cubesat = cubesat
        # reset the cubesat internal state of cubesat
        self.cubesat.reset()

        self.max_step = max_step
        self.initial_state = initial_state

        """
        Define all the external actors on the cubesat including magnetic field, solar radiation pressure, air drag,
        gravitational gradient torque
        """
        # functions that take in a state and returns a force vector <Fx,Fy,Fz> acting on the COM of the cubesat
        self.forces = []
        # functions that take in a state and returns an acceleration vector <ax,ay,az> acting on the COM of the cubesat
        self.accelerations = []
        # function that takes in a state and returns a torque vector <tx, ty, tz> acting around the COM of the
        # cubesat with inertia I
        self.torques = []

    def compute_external_force(self, s):

        net_force = np.zeros(3)
        for forcer in self.forces:
            net_force += forcer(s, self.cubesat, self.planet)

        net_acc = np.zeros(3)
        for accelerator in self.accelerations:
            net_acc += accelerator(s, self.cubesat, self.planet)

        net_force += net_acc * self.cubesat.mass

        return net_force

    def compute_external_torque(self, s):

        net_torque = np.zeros(3)
        for torquer in self.torques:
            net_torque += torquer(s, self.cubesat, self.planet)

        return net_torque

    def sample(self, t, y):
        """
        Takes in the time and state vector y and returns the derivative at that point
        to be used by an integrate such as scipy ivp_solver.

        State vector: (x, y, z, dx, dy, dz, roll, pitch, yaw, droll, dpitch, dyaw)
        """
        # print progress update if correct time has elapsed
        if t - self.last_t > 10:
            print("\rSimulation progress: {:.2f}%".format(t / (self.stop - self.start) * 100), end="")

        external_state = State.state_from_vector(y)

        # 1. Compute the external forces acting on the cubesat
        external_force = self.compute_external_force(external_state)
        external_torque = self.compute_external_torque(external_state)

        # 2. Compute the internal / commanded forces / torques
        internal_force, internal_torque = self.cubesat(t, external_state, external_force, external_torque,
                                                       self.planet.get_magnetic_field(external_state))
        internal_force = external_state.get_orientation().apply(internal_force, inverse=True)

        # 3. Sum external and internal forces / torques
        net_force, net_torque = external_force + internal_force, external_torque + internal_torque

        # 4. Get the net acceleration
        net_acc = net_force / self.cubesat.mass

        # 5. Quaternion computations for integrating orientation and find the net angular acceleration in body frame
        angular_velocity = external_state.get_angular_velocity_vector()
        angular_momentum = np.dot(self.cubesat.inertia, angular_velocity)
        angular_acc = np.dot(self.cubesat.inertia_inv, (net_torque - np.cross(angular_velocity, angular_momentum)))

        return (*external_state.get_velocity_vector(),
                *net_acc,
                *external_state.get_quaternion_derivative(),
                *angular_acc)

    def run(self, start=0, stop=5000, sample_resolution=10):
        """
        Computes the states from the current time step at t start to the state a t stop using
         Runge-Kutta 45 (ode45) with scipy ivp.
        Returns
        -------
        tuple
            A tuple that contains the time steps of each state and the states themselves
        """

        # Good reference: https://www.marksmath.org/visualization/orbits/CentralOrbit.html
        # we set atol and rtol to large values so that the max step size is always taken
        # Note that default values are 1e-3 for rtol and 1e-6 for atol.
        self.stop, self.start, self.last_t = stop, start, start
        sol = integrate.solve_ivp(
            self.sample,
            (start, stop),
            self.initial_state.state_vector,
            t_eval=np.linspace(start, self.stop, (self.stop - self.start) * sample_resolution),
            max_step=self.max_step,
            atol=1e10,
            rtol=1e10)
        print("\n---")

        # Convert state arrays into state objects for easier manipulation
        states = [State.state_from_vector(y) for y in sol.y.T]
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


class Planet:
    def __init__(self, mass, radius, magnetic_field_model=pyIGRF):
        assert isinstance(mass, (int, float)) and mass > 0, "mass must be a positive value"
        assert isinstance(radius, (int, float)) and radius > 0, "radius must be a positive value"
        self.radius = radius  # m
        self.mass = mass  # kg
        self.magnetic_field_model = magnetic_field_model

    def get_gravitational_acceleration(self, state):
        assert isinstance(state, State), "state must be a State object"
        r_hat = state.state_vector[:3] / np.linalg.norm(state.state_vector[:3])
        a = -constants.G * self.mass / np.linalg.norm(state.state_vector[:3]) ** 2 * r_hat
        return a

    def get_magnetic_field(self, state, ecef=True):
        assert isinstance(state, State), "state must be a State object"
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
            state.get_lat(), state.get_lon(), (state.get_r() - constants.R_EARTH) / 1000, 1999)
        magnetic_field_vector = np.array([s[3], s[4], s[5]]) / 1e9  # converted to T
        if ecef:
            return state.ned_to_ecef(magnetic_field_vector)
        return state
