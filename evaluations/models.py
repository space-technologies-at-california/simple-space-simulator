import numpy as np
import scipy.spatial.transform as transform

import simple_space_simulator.cubesat as cubesat


class CubeSpaceRod(cubesat.ControlDevice):
    def __init__(self, position, orientation, id, cubesat, number_of_loops, area, mu=(4 * np.pi) * (10 ** -7)):
        super().__init__(position, orientation, id, cubesat)

        self.max_I =
        self.R = transform.Rotation.from_euler('xyz', self.orientation)

    def command(self, time, command):
        # Gets its I by looking at the control values
        magnitude_m = command['I'] *
        m = self.R.apply([1, 0, 0]) * magnitude_m
        return np.zeros(3), np.zeros(3), m


class CubeSpaceCoil(cubesat.ControlDevice):
    def __init__(self, position, orientation, id, cubesat, number_of_loops, area, mu=(4 * np.pi) * (10 ** -7)):
        super().__init__(position, orientation, id, cubesat)

        self.N = number_of_loops  # Number of windings of the coil
        self.S = area
        self.mu = mu  # magnetic permeability of the core

        self.R = transform.Rotation.from_euler('xyz', self.orientation)

    def command(self, time, command):
        # Gets its I by looking at the control values
        magnitude_m = command['I'] * self.N * self.S
        m = self.R.apply([1, 0, 0]) * magnitude_m
        return np.zeros(3), np.zeros(3), m


class CubeSpaceSmallRod(CubeSpaceRod):
    pass


class CubeSpaceMediumRod(CubeSpaceRod):
    pass


class CubeSpaceLargeRod(CubeSpaceRod):
    pass


class CubeSpaceDoubleCoil(CubeSpaceCoil):
    pass


class ScheduledController(cubesat.Controller):
    """
    Detumble the cubesat until its angular velocity goes below a certain threshold and then
    enter into a PID control mode that will attempt to maximize the amount of time spent pointing toward the earth.
    Pointing to earth will be achieved by orienting the satelite such that its  z axis aligns with the
    acceleration vector of the satellite. Setpoint is thus: (0, 0, Gm)

    Additional state and state transitions can be added to the FSM to allow for accelerations up to a certain
    angular velocity and back down
    """

    def __init__(self, gain, na, max_current=0.5):
        super().__init__()
        self.k = gain
        # Number of loops times the cross sectional area of the magnetorquer coil
        self.na = na
        self.max_current = max_current

    def __call__(self, time, internal_state, sensor_readings):
        # return controller command dictionary ex. {'m0': {'I': 10}, 'm1': {'I', 2}, ...}
        # B-dot implementation
        b_dot = np.cross(internal_state['angular velocity'], internal_state['magnetic field'])
        m_desired = (self.k / np.linalg.norm(internal_state['magnetic field'])) * b_dot
        ix, iy, iz = np.clip(m_desired / self.na, a_max=self.max_current, a_min=-self.max_current)
        command = {'mx': {'I': ix}, 'my': {'I': iy}, 'mz': {'I': iz}}
        return command
