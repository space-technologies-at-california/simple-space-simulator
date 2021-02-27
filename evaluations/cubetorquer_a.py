"""
https://www.cubespace.co.za/products/adcs-components/cubetorquer/#cubetorquer-downloads

2x CubeTorquer Smalls
1x CubeTorquer Coil
"""

import numpy as np

import simple_space_simulator.physics as physics
from simple_space_simulator import constants
import simple_space_simulator.cubesat as cubesat
import simple_space_simulator.renderer as renderer
import simple_space_simulator.plots as plots
import simple_space_simulator.utils as utils
import simple_space_simulator.components as components
import simple_space_simulator.state as state

from evaluations.models import CubeSpaceSmallRod, CubeSpaceMediumRod, CubeSpaceLargeRod, CubeSpaceCoil, \
    ScheduledController


# helpful ballpark k calculator to prevent too much railing, make sure to still tune for power consumption
def bdot_gain_calculator(magnetic_gain, max_current, max_angular_velocity):
    return max_current * magnetic_gain / max_angular_velocity


"""
Step 1: Define the cubesat and planet specifications
"""
# Define the magnetorquer properties
magnetorquer_x = CubeSpaceSmallRod(position=(0, 0, 0), orientation=(0, 0, 0), id='mx', max_voltage=5)
magnetorquer_y = CubeSpaceSmallRod(position=(0, 0, 0), orientation=(0, 0, np.pi / 2), id='my', max_voltage=5)
magnetorquer_z = CubeSpaceCoil(position=(0, 0, 0), orientation=(0, -np.pi / 2, 0), id='mz', max_voltage=5)

# Build the controller
max_currents = np.array([magnetorquer_x.get_max_current(),
                         magnetorquer_y.get_max_current(),
                         magnetorquer_z.get_max_current()])
magnetic_gains = np.array([magnetorquer_x.magnetic_gain,
                           magnetorquer_y.magnetic_gain,
                           magnetorquer_z.magnetic_gain])

max_angular_vel = 0.1
print(f'Suggested maximum bdot gains:\n'
      f'x: {bdot_gain_calculator(magnetorquer_x.magnetic_gain, magnetorquer_x.get_max_current(), max_angular_vel)}\n'
      f'y: {bdot_gain_calculator(magnetorquer_y.magnetic_gain, magnetorquer_y.get_max_current(), max_angular_vel)}\n'
      f'z: {bdot_gain_calculator(magnetorquer_z.magnetic_gain, magnetorquer_z.get_max_current(), max_angular_vel)}\n')

controller = ScheduledController(gains=np.array([4.2, 4.2, 1.27]),
                                 magnetic_gains=magnetic_gains,
                                 max_currents=max_currents)

# mass in kg, length, width, height in m
qubesat = cubesat.Cubesat(mass=1, controller=controller,
                          state_estimator=components.SimpleStateEstimator(),
                          length=0.2, width=0.2, height=0.4)

# Define the components that will be added to the satellite
lsm9ds1 = components.SimpleIMU(position=(0, 0, 0), orientation=(0, 0, 0), id='imu', cubesat=qubesat)

qubesat.add_sensor(lsm9ds1)
qubesat.add_actuator(magnetorquer_x)
qubesat.add_actuator(magnetorquer_y)
qubesat.add_actuator(magnetorquer_z)

# dipole in tesla referenced from the cubesat frame
qubesat.add_static_magnetic_dipole(np.array([0.0, 0, 0]))

# define planet specification
planet = physics.Planet(constants.M_EARTH, constants.R_EARTH)  # mass in kg, radius in meters

"""
Step 2: Configure the initial state of the cubesat in the simulation
"""
inclination = constants.ISS_INCLINATION
altitude = constants.ISS_ALTITUDE
max_step_size = 100

print("Starting inclination:", str(np.degrees(inclination)) + "deg", "\nStarting altitude:", str(altitude) + "m", '\n')

v_init = utils.inclination_to_cartesian_velocity(utils.circular_orbit_velocity(altitude), inclination)

# roll, pitch, yaw are defined referenced to ecef, droll, dpitch, dyaw are body rates
# r, p, y is converted to quaternion with the following functions. pqr are angular velocities
q_init = utils.euler_to_quaternion(0, 0, 0)
pqr_init = [0.1, 0.1, 0.1]

initial_state = state.State(altitude + constants.R_EARTH, 0, 0, *v_init, *q_init, *pqr_init)
simulator = physics.Simulator(qubesat, planet, initial_state, max_step_size)

"""
Step 3: Add all the desired force, torque, and acceleration functions to the simulator
inputs are <state, cubesat, planet>
"""
# Acceleration due to the planet
simulator.add_accelerator(lambda s, c, p: planet.get_gravitational_acceleration(s))

"""
Step 4: Write the state estimation and control objects
"""

"""
Step 5: Configure the stop condition for the simulation. Run the simulation with the desired renderer
"""
num_orbits = 15
r = renderer.Renderer(resolution=1)
r.run(simulator, stop_time=int(num_orbits * utils.orbital_period(altitude)))

"""
Step 6: Choose the plots you want to display after running the simulation
"""
# plot1 = plots.CartesianPlot()
# r.add_plot(plot1)
# plot2 = plots.CartesianVelocityPlot()
# r.add_plot(plot2)
# plot3 = plots.SphericalPlot()
# r.add_plot(plot3)
# plot4 = plots.SphericalVelocityPlot()
# r.add_plot(plot4)
# plot5 = plots.OrbitalPlot2D(planet, inclination=inclination)
# r.add_plot(plot5)
# plot6 = plots.OrbitalPlot3D(planet, show_magnetic_field=True, show_planet=True)
# r.add_plot(plot6)
# plot7 = plots.OrientationPlot()
# r.add_plot(plot7)
plot8 = plots.AngularVelocityPlot(qubesat)
r.add_plot(plot8)
# plot9 = plots.MagneticFieldPlot(planet)
# r.add_plot(plot9)
plot10 = plots.MagnetorquerCurrentPlot(qubesat)
r.add_plot(plot10)

"""
Step 7: Display all the plots
"""
r.render(figsize=(20, 7), columns=2)

"""
Step 8: Run any animated plots
"""
# animated_plot1 = plots.OrientationPlotAnimated(qubesat, planet, rtf_multiplier=20)
# r.run_animated_plot(animated_plot1, 10.0, start_time=0, stop_time=r.time_stamps[-1])

"""
Step 9: Save the data
"""
r.save('cubetorquer_a.npy')
