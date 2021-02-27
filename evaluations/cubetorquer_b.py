"""
https://www.cubespace.co.za/products/adcs-components/cubetorquer/#cubetorquer-downloads

3x CubeTorquer Smalls
"""

import simple_space_simulator.cubesat as cubesat


class SimpleSolenoidMagnetorquer(cubesat.ControlDevice):
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

class SimpleController(cubesat.Controller):
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