import numpy as np
import simple_space_simulator.utils as utils
import simple_space_simulator.cubesat as cubesat

# constants
class Consts:
    G = 6.67430e-11 # m^3/(kg*s^2)
    M_earth = 5.972e24
    R_earth = 6.371e6
    ISS_altitude = 4.08e6
    ISS_inclination = 0.9005899
    mu = G * M_earth
    miles_to_meters = 1609.34


class Simulator:
    def __init__(self, cubesat, planet, state, dt=0.1):
        """
        time and step size in seconds
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
        # compute acceleration vector from gravity assuming points mass of the earth and satelite
        # [x,y,z]/|[x,y,z]| <- assumes point mass earth with center at (0,0,0)

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
        w_norm = np.linalg.norm(w)
        w_f = w * np.sin(w_norm / 2) / w_norm if w_norm != 0 else [0, 0, 0]
        q = utils.quaternion_multiply(np.array([np.cos(w_norm / 2), w_f[0], w_f[1], w_f[2]]),
                                self.state.get_orientation_quaternion())
        q /= np.linalg.norm(q)

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
    def __init__(self, mass, radius):
        """
        mass in kg
        radius in m
        """
        self.radius = radius
        self.mass = mass
    
    def get_gravitational_acceleration(self, state):
        r_hat = state.state_vector[:3]/np.linalg.norm(state.state_vector[:3])
        a = -Consts.G * self.mass / np.linalg.norm(state.state_vector[:3])**2 * r_hat
        return a
