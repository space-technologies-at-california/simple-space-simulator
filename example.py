import numpy as np

import simple_space_simulator.physics as physics
from simple_space_simulator import constants
import simple_space_simulator.cubesat as cubesat
import simple_space_simulator.renderer as renderer
import simple_space_simulator.plots as plots
import simple_space_simulator.utils as utils

"""
Step 1: Define the cubesat and planet specifications
"""
qubesat = cubesat.Cubesat(1, length=0.2, width=0.2, height=0.4)  # mass in kg, length, width, height in m

static_dipole = np.array([0, 0, 0])  # dipole in tesla referenced from the cubesat frame
qubesat.add_magnetic_dipole(static_dipole)
planet = physics.Planet(constants.M_EARTH, constants.R_EARTH)  # mass in kg, radius in meters

"""
Step 2: Configure the initial state of the cubesat in the simulation
"""
inclination = 0
step_size = 10

vx, vy, vz = utils.inclination_to_cartesian_velocity(utils.circular_orbit_velocity(constants.ISS_ALTITUDE), inclination)
qw, qx, qy, qz = utils.euler_to_quaternion(0, 0, 0)
# x, y, z, dx, dy, dz, qw, qx, qy, qz, wx, wy, wz
state = cubesat.State(constants.ISS_ALTITUDE + constants.R_EARTH, 0, 0, vx, vy, vz, qw, qx, qy, qz, 0.001, 0.0, 0.0)
simulator = physics.Simulator(qubesat, planet, state, step_size)

"""
Step 3: Add all the desired force, torque, and acceleration functions to the simulator
inputs are <state, cubesat, planet>
"""
# Acceleration due to the planet
simulator.add_accelerator(lambda s, c, p: planet.get_gravitational_acceleration(s))


# Angular torque due to the magnetic field
# https://docs.google.com/document/d/1H6u0sJonnc3Cq24zajG50wIkeIDk1tscy8mXIZIa8WQ/edit
def magnetic_torques(s, c, p):
    field = p.get_magnetic_field(s)
    t = np.cross(c.get_magnetic_dipole(s), field)
    return t


simulator.add_torquer(magnetic_torques)

"""
Step 4: Configure the stop condition for the simulation. Run the simulation with the desired renderer
"""


# stop after a specified number of steps
def stop_condition(states, times): return len(states) > utils.steps_per_orbit(constants.ISS_ALTITUDE, step_size)


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
plot5 = plots.OrbitalPlot2D(planet, inclination=inclination)
r.add_plot(plot5)
plot6 = plots.OrbitalPlot3D(planet, show_magnetic_field=True, show_planet=False)
r.add_plot(plot6)
plot7 = plots.OrientationPlot()
r.add_plot(plot7)
plot8 = plots.AngularVelocityPlot()
r.add_plot(plot8)
plot9 = plots.MagneticFieldPlot(planet)
r.add_plot(plot9)

"""
Step 6: Display all the plots
"""
r.render(figsize=(5, 7), columns=4)

"""
Step 7: Run any animated plots
"""
animated_plot1 = plots.OrientationPlotAnimated(qubesat)
# r.run_animated_plot(animated_plot1, 20.0, start_time=0, stop_time=500)
