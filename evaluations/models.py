import numpy as np
import scipy.spatial.transform as transform

import simple_space_simulator.cubesat as cubesat


class CubeSpaceMagnetorquer(cubesat.ControlDevice):
    def __init__(self, position, orientation, id, magnetic_gain, max_current, max_voltage, resistance):
        super().__init__(position, orientation, id)

        self.magnetic_gain = magnetic_gain  # A*m^2/A

        # Used to make sure that the supplied voltage is feasible
        self.max_voltage = max_voltage
        self.resistance = resistance

        # maximum current as to the specification, this is for safety check
        self.max_current = max_current

        self.R = transform.Rotation.from_euler('xyz', self.orientation)

    def get_max_current(self):
        return min(self.max_current, self.max_voltage / self.resistance)

    def command(self, time, command):
        i_max = self.get_max_current()
        if -i_max <= command['I'] <= i_max:
            magnitude_m = command['I'] * self.magnetic_gain
        else:
            raise Exception(f"Magnetorquer over current! Current magnitude was {abs(command['I'])} "
                            f"but should be less than {i_max}")
        m = self.R.apply([1, 0, 0]) * magnitude_m
        return np.zeros(3), np.zeros(3), m


class CubeSpaceSmallRod(CubeSpaceMagnetorquer):
    def __init__(self, position, orientation, id, max_voltage=5):
        super().__init__(position, orientation, id, 2.8, 150e-3, max_voltage, 31)


class CubeSpaceMediumRod(CubeSpaceMagnetorquer):
    def __init__(self, position, orientation, id, max_voltage=5):
        super().__init__(position, orientation, id, 8.2, 150e-3, max_voltage, 65)


class CubeSpaceLargeRod(CubeSpaceMagnetorquer):
    def __init__(self, position, orientation, id, max_voltage=5):
        super().__init__(position, orientation, id, 25, 150e-3, max_voltage, 66)


class CubeSpaceCoil(CubeSpaceMagnetorquer):
    def __init__(self, position, orientation, id, max_voltage=5):
        super().__init__(position, orientation, id, 2.1, 150e-3, max_voltage, 83)


class ScheduledController(cubesat.Controller):
    """
    Detumble the cubesat until its angular velocity goes below a certain threshold and then
    enter into a PID control mode that will attempt to maximize the amount of time spent pointing toward the earth.
    Pointing to earth will be achieved by orienting the satelite such that its  z axis aligns with the
    acceleration vector of the satellite. Setpoint is thus: (0, 0, Gm)

    Additional state and state transitions can be added to the FSM to allow for accelerations up to a certain
    angular velocity and back down
    """

    def __init__(self, gains, max_currents, magnetic_gains):
        super().__init__()
        self.k = gains
        self.max_currents = max_currents
        self.magnetic_gains = magnetic_gains

    def __call__(self, time, internal_state, sensor_readings):
        # return controller command dictionary ex. {'m0': {'I': 10}, 'm1': {'I', 2}, ...}
        # B-dot implementation
        b_dot = np.cross(internal_state['angular velocity'], internal_state['magnetic field'])
        m_desired = (self.k / np.linalg.norm(internal_state['magnetic field'])) * b_dot
        ix, iy, iz = np.clip(m_desired / self.magnetic_gains, a_max=self.max_currents, a_min=-self.max_currents)
        command = {'mx': {'I': ix}, 'my': {'I': iy}, 'mz': {'I': iz}}
        return command
