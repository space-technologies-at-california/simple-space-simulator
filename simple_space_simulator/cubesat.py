import numpy as np
from abc import ABC, abstractmethod

import simple_space_simulator.utils as utils


class SensorDevice(ABC):
    ids = []

    def __init__(self, position, orientation, id, cubesat):
        assert id not in SensorDevice.ids, "sensor device id must be unique"

        # position is x y z from center of cubesat
        # orientation is r p y in cubesate frame
        self.position = position
        self.orientation = orientation
        self.id = id
        self.cubesat = cubesat
        SensorDevice.ids.append(self.id)

    @abstractmethod
    def read(self, time, external_state, external_linear_acc, external_angular_acc, magnetic_field):
        # return dictionary of values with ids ex. {'ax': 5, 'ay': 6, ...}
        pass

    def reset(self):
        pass


class ControlDevice(ABC):
    ids = []

    def __init__(self, position, orientation, id, cubesat):
        assert id not in ControlDevice.ids, "control device id must be unique"
        # position is x y z from center of cubesat
        # orientation is r p y in cubesate frame
        self.position = position
        self.orientation = orientation
        self.id = id
        self.cubesat = cubesat
        ControlDevice.ids.append(self.id)

    @abstractmethod
    def command(self, time, control_command):
        # return actuator_force, actuator_torque, actuator_dipole
        pass

    def reset(self):
        pass


class Controller(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, time, internal_state, sensor_readings):
        # return controller command dictionary ex. {'m0': {'I': 10}, 'm1': {'I', 2}, ...}
        pass

    def reset(self):
        pass


class StateEstimator(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, time, sensor_readings):
        # return internal state vector [ax, ay, az, aax, aay, aaz, mx, my, mz, ...]
        pass

    def reset(self):
        pass


class Cubesat:
    def __init__(self, mass, controller, state_estimator, length=0.1, width=0.1, height=0.1,
                 inertia=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])):
        """
        mass in kg
        length, width, height in m
        """
        assert isinstance(mass, (int, float)) and mass > 0, "mass must be a positive value"
        assert isinstance(length, (int, float)) and length > 0, "length must be a positive value"
        assert isinstance(width, (int, float)) and width > 0, "width must be a positive value"
        assert isinstance(height, (int, float)) and height > 0, "height must be a positive value"
        assert isinstance(inertia, np.ndarray) and inertia.ndim == 2, "inertia must be a 2d matrix"

        self.mass = mass
        self.inertia = inertia
        self.inertia_inv = np.linalg.inv(inertia)

        # Static magnetic dipoles as vectors in 3d
        self.static_magnetic_dipoles = []

        # Actuators
        self.actuators = []

        # Sensor dictionary
        self.sensors = []

        # Control logic
        self.controller = controller

        # State estimation
        self.state_estimator = state_estimator

        # Dimensions
        self.length = length
        self.width = width
        self.height = height
        self.points = np.array([[-width / 2, -length / 2, -height / 2],
                                [width / 2, -length / 2, -height / 2],
                                [width / 2, length / 2, -height / 2],
                                [-width / 2, length / 2, -height / 2],
                                [-width / 2, -length / 2, height / 2],
                                [width / 2, -length / 2, height / 2],
                                [width / 2, length / 2, height / 2],
                                [-width / 2, length / 2, height / 2]])

        # Variables for analysis and rendering
        self.internal_state_history = []
        self.control_history = []

    def __call__(self, time, external_state, external_linear_acc, external_angular_acc, external_magnetic_field):
        external_magnetic_field += self.get_static_magnetic_dipole()

        sensor_readings = {}
        for sensor in self.sensors:
            id = sensor.id
            sensor_readings[id] = sensor.read(
                time, external_state, external_linear_acc, external_angular_acc, external_magnetic_field)

        internal_state = self.state_estimator(time, sensor_readings)
        self.internal_state_history.append(internal_state)
        control_commands = self.controller(time, internal_state, sensor_readings)

        internal_force, internal_torque = np.zeros(3), np.zeros(3)
        magnetic_dipole = self.get_static_magnetic_dipole()
        for actuator in self.actuators:
            if control_commands is not None and actuator.id in control_commands:
                command = control_commands[actuator.id]
                actuator_force, actuator_torque, actuator_dipole = actuator.command(time, command)
                internal_force += actuator_force
                internal_torque += actuator_torque
                magnetic_dipole += actuator_dipole

        # incorporate torque due to magnetic field
        internal_torque += np.cross(utils.quaternion_rotate(external_state.get_orientation_quaternion(),
                                                            magnetic_dipole),
                                    external_magnetic_field)

        self.control_history.append({'force': internal_force,
                                     'torque': internal_torque,
                                     'magnetic dipole': magnetic_dipole,
                                     'actuator commands': control_commands})

        return internal_force, internal_torque

    def reset(self):
        self.internal_state_history = []
        self.control_history = []
        self.controller.reset()
        self.state_estimator.reset()
        for actuator in self.actuators:
            actuator.reset()
        for sensor in self.sensors:
            sensor.reset()
        print("Cubesat reset")

    def add_static_magnetic_dipole(self, dipole):
        assert isinstance(dipole, np.ndarray) and dipole.ndim == 1, "dipole must be a vector"
        self.static_magnetic_dipoles.append(dipole)

    def add_actuator(self, actuator):
        self.actuators.append(actuator)

    def add_sensor(self, sensor):
        self.sensors.append(sensor)

    def get_static_magnetic_dipole(self):
        dipole_sum = np.array([0.0, 0.0, 0.0])
        for dipole in self.static_magnetic_dipoles:
            # dipole_sum += utils.quaternion_rotate(s.get_orientation_quaternion(), dipole)
            dipole_sum += dipole
        return dipole_sum
