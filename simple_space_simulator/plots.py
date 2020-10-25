import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib.ticker import FuncFormatter, MultipleLocator
import numpy as np
import math

class CartesianPlot:
    def __init__(self):
        self.is_3d = False
    
    def build(self, states, time_stamps, ax):
        x = [state.get_x() for state in states]
        y = [state.get_y() for state in states]
        z = [state.get_z() for state in states]
        ax.plot(time_stamps, x, color="red", label='x')
        ax.plot(time_stamps, y, color="green", label='y')
        ax.plot(time_stamps, z, color="blue", label='z')
        ax.set_title("Cartesian Coordinates Plot")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("position (m)")
        
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels)
        
class CartesianVelocityPlot:
    def __init__(self):
        self.is_3d = False
    
    def build(self, states, time_stamps, ax):
        dx = [state.get_dx() for state in states]
        dy = [state.get_dy() for state in states]
        dz = [state.get_dz() for state in states]
        ax.plot(time_stamps, dx, color="red", label='dx')
        ax.plot(time_stamps, dy, color="green", label='dy')
        ax.plot(time_stamps, dz, color="blue", label='dz')
        ax.set_title("Cartesian Velocity Plot")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("velocity (m/s)")
        
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels)


class SphericalPlot:
    def __init__(self):
        self.is_3d = False
    
    def build(self, states, time_stamps, ax):
        r = [state.get_r() for state in states]
        lon = [state.get_lon() for state in states]
        lat = [state.get_lat() for state in states]
        lns1 = ax.plot(time_stamps, lat, color="green", label='lat')
        lns2 = ax.plot(time_stamps, lon, color="blue", label='lon')
        ax.set_title("Spherical Coordinates Plot")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("angle (radians)")
        ax.yaxis.set_major_locator(MultipleLocator(base=np.pi/2))
        
        ax2 = ax.twinx()
        lns3 = ax2.plot(time_stamps, r, color="red", label='radius')
        
        lns = lns1+lns2+lns3
        labels = [l.get_label() for l in lns]
        ax.legend(lns, labels, loc='upper center')
        
class SphericalVelocityPlot:
    def __init__(self):
        self.is_3d = False
    
    def build(self, states, time_stamps, ax):
        r = [state.get_dr() for state in states]
        lon = [state.get_dtheta() for state in states]
        lat = [state.get_dphi() for state in states]
                
        lns1 = ax.plot(time_stamps, lat, color="green", label='dphi')
        lns2 = ax.plot(time_stamps, lon, color="blue", label='dtheta')
        ax.set_title("Spherical Velocity Plot")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("angle (radians) / s")
        ax.yaxis.set_major_locator(MultipleLocator(base=np.pi/2))
        
        ax2 = ax.twinx()
        lns3 = ax2.plot(time_stamps, r, color="red", label='dr')
        
        lns = lns1+lns2+lns3
        labels = [l.get_label() for l in lns]
        ax.legend(lns, labels)

# The following plots should take in a planet for determining radius. Earth should be updated to an ellipse with a = 6,378,137 m ; semi-minor: b = 6,356,752.3142 m. 
# this needs to be accounted for in computing altitude and gravitational acceleration
class OrbitalPlot3D:
    def __init__(self, planet):
        self.is_3d = True
        self.planet = planet
    
    def build(self, states, time_stamps, ax):
        
        count = 15 # keep 180 points along theta and phi
        # define a grid matching the map size, subsample along with pixels
        theta = np.linspace(0, np.pi, count)
        phi = np.linspace(0, 2*np.pi, count)
        theta,phi = np.meshgrid(theta, phi)
        R = self.planet.radius
        # sphere
        x = R * np.sin(theta) * np.cos(phi)
        y = R * np.sin(theta) * np.sin(phi)
        z = R * np.cos(theta)
        ax.plot_surface(x.T, y.T, z.T, cstride=1, rstride=1) # we've already pruned ourselves
        
        x = [state.get_x() for state in states]
        y = [state.get_y() for state in states]
        z = [state.get_z() for state in states]
        ax.plot3D(x, y, z, 'red')
        ax.set_title("Orbital Plot 3D")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        
        ax.set_xlim(-1.5e7, 1.5e7)
        ax.set_ylim(-1.5e7, 1.5e7)
        ax.set_zlim(-1.5e7, 1.5e7)


        
class OrbitalPlot2D:
    def __init__(self, planet, inclination=0):
        self.is_3d = False
        self.planet = planet
        self.inclination = inclination
    
    def build(self, states, time_stamps, ax):
        
        ax.add_artist(plt.Circle((0,0), self.planet.radius, color='blue'))
 
        x = [state.get_x() for state in states]
        y = [state.get_y() for state in states]
        z = [state.get_z() for state in states]
        
        T = np.array([[1, 0, 0],
                       [0, math.cos(-self.inclination), -math.sin(-self.inclination)],
                       [0, math.sin(-self.inclination), math.cos(-self.inclination)]]) 
        for i in range(len(x)):
            p = np.dot(T, np.array([x[i],y[i],z[i]]))
            x[i] = p[0]
            y[i] = p[1]
            z[i] = p[2]
        
        ax.plot(x, y, color="red", label='x')
        ax.set_title("Orbital Plot 2D")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(-1.5e7, 1.5e7)
        ax.set_ylim(-1.5e7, 1.5e7)