import math

import simple_space_simulator.physics as physics
from simple_space_simulator import constants
import simple_space_simulator.cubesat as cubesat
import simple_space_simulator.renderer as renderer
import simple_space_simulator.plots as plots
import simple_space_simulator.utils as utils

"""
Step 1: Define the cubesat and planet specifications
"""
qubesat = cubesat.Cubesat(1, length=0.1, width=0.1, height=0.1)  # mass in kg, length, width, height in m
planet = physics.Planet(constants.M_EARTH, constants.R_EARTH)  # mass in kg, radius in meters

"""
Step 2: Configure the initial state of the cubesat in the simulation
"""
vx, vy, vz = utils.inclination_to_cartesian_velocity(utils.circular_orbit_velocity(constants.ISS_ALTITUDE), math.pi / 2)
qw, qx, qy, qz = utils.euler_to_quaternion(0, 0, 0)
# x, y, z, dx, dy, dz, qw, qx, qy, qz, wx, wy, wz
state = cubesat.State(constants.ISS_ALTITUDE + constants.R_EARTH, 0, 0, vx, vy, vz, qw, qx, qy, qz, 0.001,
                      0.000, 0.000)
simulator = physics.Simulator(qubesat, planet, state, 1)

"""
Step 3: Add all the desired force, torque, and acceleration functions to the simulator
"""
simulator.add_accelerator(lambda s, c, p: planet.get_gravitational_acceleration(s))  # Acceleration due to the planet

"""
Step 4: Configure the stop condition for the simulation. Run the simulation with the desired renderer
"""


# stop after a specified number of steps
def stop_condition(states, times): return len(states) > 10000


r = renderer.Renderer()
r.run(simulator, stop_condition)

"""
Step 5: Choose the plots you want to display after running the simulation
"""
plot1 = plots.CartesianPlot()
r.add_plot(plot1)
plot2 = plots.CartesianVelocityPlot()
r.add_plot(plot2)
plot3 = plots.SphericalPlot()
r.add_plot(plot3)
plot4 = plots.SphericalVelocityPlot()
r.add_plot(plot4)
plot5 = plots.OrbitalPlot2D(planet, inclination=constants.ISS_INCLINATION)
r.add_plot(plot5)
plot6 = plots.OrbitalPlot3D(planet)
r.add_plot(plot6)
plot7 = plots.OrientationPlot()
r.add_plot(plot7)

"""
Step 6: Display all the plots
"""
r.render(figsize=(15, 25), columns=2)
