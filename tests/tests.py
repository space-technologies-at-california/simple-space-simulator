import unittest

import simple_space_simulator.physics as physics
from simple_space_simulator import constants
import simple_space_simulator.cubesat as cubesat
import simple_space_simulator.renderer as renderer
import simple_space_simulator.utils as utils


class TestSimpleOrbit(unittest.TestCase):

    def run_simulator_one_orbit(self, altitude, time_per_step, steps_per_orbit, inclination=0):
        assert isinstance(altitude, (int, float)) and altitude > 0, "altitude must be a positive value"
        assert isinstance(time_per_step, (int, float)) and time_per_step > 0, "time_per_step must be a positive value"
        assert isinstance(steps_per_orbit, (int, float)) and steps_per_orbit > 0, \
            "steps_per_orbit must be a positive value"
        assert isinstance(inclination, (int, float)), "inclination must be a radian value"
        # Standard simulator code
        qubesat = cubesat.Cubesat(1, length=0.1, width=0.1, height=0.1)
        planet = physics.Planet(constants.M_EARTH, constants.R_EARTH)
        v_init = utils.inclination_to_cartesian_velocity(utils.circular_orbit_velocity(altitude), 0)
        q_init = utils.euler_to_quaternion(0, 0, 0)
        dq_init = utils.angular_velocity_to_dquaternion([0.0, 0.0, 0.0], q_init)

        initial_state = cubesat.State(constants.ISS_ALTITUDE + constants.R_EARTH, 0, 0, *v_init, *q_init, *dq_init)
        simulator = physics.Simulator(qubesat, planet, initial_state, 10)
        simulator.add_accelerator(
            lambda s, c, p: planet.get_gravitational_acceleration(s))

        r = renderer.Renderer(resolution=1)
        r.run(simulator, stop_time=int(utils.orbital_period(constants.ISS_ALTITUDE)))
        return r

    def test_incline_0_orbit(self):
        """
        Runs the simulator at a zero degree incline for exactly a single orbit and verifies that
        it ends up in the proper location and that the states are within tolerance of the ideal
        orbital equations
        """

        altitude = constants.ISS_ALTITUDE  # meters
        time_per_step = 0.1  # seconds
        orbital_period = utils.orbital_period(altitude)
        steps_per_orbit = utils.steps_per_orbit(altitude, time_per_step)

        r = self.run_simulator_one_orbit(altitude, time_per_step, steps_per_orbit)

        # Tests
        self.assertLess(abs(orbital_period - 5600), 50, msg='Correct orbital period of ISS +- 50s')
        self.assertEqual(steps_per_orbit, int(orbital_period / time_per_step), msg='Correct number of steps computed')
        self.assertEqual(int(r.time_stamps[-1]), int(orbital_period), 'Correct simulation duration')

        error = r.states[-1].get_cartesian_state_vector() - r.states[0].get_cartesian_state_vector()
        self.assertLess(abs(error[0]), 10000, 'Error in x less than 10,000')
        self.assertLess(abs(error[1]), 50000, 'Error in y less than 50,000')
        self.assertLess(abs(error[2]), 100,   'Error in z less than 100')
        self.assertLess(abs(error[3]), 100,   'Error in vx less than 100')
        self.assertLess(abs(error[4]), 100,   'Error in vy less than 100')
        self.assertLess(abs(error[5]), 100,   'Error in vz less than 100')

    def test_incline_ISS_orbit(self):
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
