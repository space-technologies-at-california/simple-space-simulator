import numpy as np

import simple_space_simulator.physics as physics
from simple_space_simulator import constants
import simple_space_simulator.cubesat as cubesat
import simple_space_simulator.renderer as renderer
import simple_space_simulator.plots as plots
import simple_space_simulator.utils as utils
import simple_space_simulator.components as components
import simple_space_simulator.state as state

"""
Step 1: Define the cubesat and planet specifications
"""
# mass in kg, length, width, height in m
qubesat = cubesat.Cubesat(mass=1, controller=components.SimpleController(0.01, 5),
                          state_estimator=components.SimpleStateEstimator(),
                          length=0.2, width=0.2, height=0.4)

# Define the components that will be added to the satellite
lsm9ds1 = components.SimpleIMU(position=(0, 0, 0), orientation=(0, 0, 0), id='imu', cubesat=qubesat)
magnetorquer_x = components.SimpleSolenoidMagnetorquer(position=(0, 0, 0), orientation=(0, 0, 0), id='mx',
                                                       cubesat=qubesat, number_of_loops=10, area=0.5)
magnetorquer_y = components.SimpleSolenoidMagnetorquer(position=(0, 0, 0), orientation=(0, 0, 1.57), id='my',
                                                       cubesat=qubesat, number_of_loops=10, area=0.5)
magnetorquer_z = components.SimpleSolenoidMagnetorquer(position=(0, 0, 0), orientation=(0, 1.57, 0), id='mz',
                                                       cubesat=qubesat, number_of_loops=10, area=0.5)

qubesat.add_sensor(lsm9ds1)
qubesat.add_actuator(magnetorquer_x)
qubesat.add_actuator(magnetorquer_y)
qubesat.add_actuator(magnetorquer_z)

# dipole in tesla referenced from the cubesat frame
qubesat.add_static_magnetic_dipole(np.array([0, 0, 0]))

# define planet specification
planet = physics.Planet(constants.M_EARTH, constants.R_EARTH)  # mass in kg, radius in meters

"""
Step 2: Configure the initial state of the cubesat in the simulation
"""
inclination = constants.ISS_INCLINATION
altitude = constants.ISS_ALTITUDE
max_step_size = 1

print("Starting inclination:", str(np.degrees(inclination)) + "deg", "\nStarting altitude:", str(altitude) + "m", '\n')

v_init = utils.inclination_to_cartesian_velocity(utils.circular_orbit_velocity(altitude), inclination)

# roll, pitch, yaw are defined referenced to ecef, droll, dpitch, dyaw are body rates
# r, p, y and dr, dp, dyaw are converted to quaternion with the following two functions
q_init = utils.euler_to_quaternion(0, 0, 0)
dq_init = utils.angular_velocity_to_dquaternion([0.001, 0.0, 0.0], q_init)

initial_state = state.State(altitude + constants.R_EARTH, 0, 0, *v_init, *q_init, *dq_init)
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
r = renderer.Renderer(resolution=1)
r.run(simulator, stop_time=int(utils.orbital_period(altitude)))

"""
Step 6: Choose the plots you want to display after running the simulation
"""
plot1 = plots.CartesianPlot()
r.add_plot(plot1)
plot2 = plots.CartesianVelocityPlot()
r.add_plot(plot2)
plot3 = plots.SphericalPlot()
r.add_plot(plot3)
plot4 = plots.SphericalVelocityPlot()
r.add_plot(plot4)
plot5 = plots.OrbitalPlot2D(planet, inclination=inclination)
r.add_plot(plot5)
plot6 = plots.OrbitalPlot3D(planet, show_magnetic_field=True, show_planet=True)
r.add_plot(plot6)
plot7 = plots.OrientationPlot()
r.add_plot(plot7)
plot8 = plots.AngularVelocityPlot(qubesat)
r.add_plot(plot8)
plot9 = plots.MagneticFieldPlot(planet)
r.add_plot(plot9)

"""
Step 7: Display all the plots
"""
r.render(figsize=(5, 7), columns=4)

"""
Step 8: Run any animated plots
"""
animated_plot1 = plots.OrientationPlotAnimated(qubesat, rtf_multiplier=10)
r.run_animated_plot(animated_plot1, 20.0, start_time=0, stop_time=r.time_stamps[-1])
