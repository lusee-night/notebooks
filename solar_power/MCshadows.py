import lusee
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.linalg import norm
from scipy.interpolate import interp1d
import os
import warnings

def luseepy_sun_AltAz(night, lat, ϕ_landing):
    ## we use trickery to get sun latitude by setting long to zero
    obs = lusee.LObservation(night, lun_lat_deg=lat, lun_long_deg=0,  deltaT_sec=15*60)
    alt, az = obs.get_track_solar('sun')
    w=np.where(alt>0)
    alt=alt[w]
    az=az[w]
    az_offset=ϕ_landing*np.ones_like(az)
    return (alt,az+az_offset)
    
def sun_vector(sun_alt, sun_az):
    """
    returns the Sun's polar coordinates given its Alt-Az
    assumes that light source is at infinity
    Note: ϕ=0 is North, ϕ=π/2 is West, ϕ=π is South, and ϕ=3π/2 is East, thus ϕ = 2π - azimuth
    Note: θ=0 is Zenith, θ=π/2 is Horizon, thus θ = π/2 - altitude
    """
    θ=np.pi/2 - sun_alt
    ϕ=2*np.pi - sun_az
    sun_vec = np.array([np.sin(θ)*np.cos(ϕ),np.sin(θ)*np.sin(ϕ),np.cos(θ)])
    
    return sun_vec

def random_point(side):
    rand_x=side*(np.random.random()-0.5)
    rand_y=side*(np.random.random()-0.5)
    rand_z=0
    
    return np.array([rand_x, rand_y, rand_z])

def to2D(point):
    x,y,z=point
    return np.array([x,y])

def antenna_axis_vector(θ, ϕ):
    axis_vec = np.array([np.sin(θ)*np.cos(ϕ),np.sin(θ)*np.sin(ϕ),np.cos(θ)])
    
    return axis_vec

def distance(a1, v1, a2, v2):
    """
    return distance between two lines defined as \vec{a}+t*\vec{b}
    """
    a1=np.array(a1)
    a2=np.array(a2)
    normal_vec=np.cross(v1,v2)
    nhat=normal_vec/norm(normal_vec)
    distance = abs(np.dot(a2-a1, nhat))
    
    return distance

def shadow_check(point, sun_vec, antenna_origin, antenna_axis_vec, antenna_radius):
    """
    calculate the 3D distance between the sun vec and the antenna
    
    return True if distance is less than antenna radius, False otherwise.
    """
    distance_to_axis = distance(point, sun_vec, antenna_origin, antenna_axis_vec)
    shadow=True if distance_to_axis<antenna_radius else False
    
    return shadow

def monte_carlo_shadow(sun_vec, side, radius, θ_antenna,ϕ_antenna, nsamples=1000):
    """
    uniformly samples the panel and checks each sampled point if it's in shadow or not
    
    returns two lists: inShadow and notInShadow containing coordinates of the sampled points
    """
    antenna_origin=np.array([0.0,0.0,0.1])
    antenna_axis_vec1=antenna_axis_vector(θ_antenna,ϕ_antenna)
    antenna_axis_vec2=antenna_axis_vector(θ_antenna,ϕ_antenna+np.pi/2)
    
    inShadow=[]
    notInShadow=[]
    for i in range(nsamples):
        point=random_point(side)

        shadowed1=shadow_check(point,sun_vec, antenna_origin, antenna_axis_vec1, antenna_radius=radius)
        shadowed2=shadow_check(point,sun_vec, antenna_origin, antenna_axis_vec2, antenna_radius=radius)
        shadowed = shadowed1 or shadowed2

        if shadowed:
            inShadow.append(to2D(point))
        else:
            notInShadow.append(to2D(point))
    
    return inShadow, notInShadow
        
def save_shadow_samples(config_name, night, lat, sun_alt, sun_az, inShadow, notInShadow):
    cwd=os.getcwd()
    path=f'/shadow_samples/night{night}_lat{lat}/'
    if not os.path.exists(cwd+path): os.makedirs(cwd+path)
    fname=f'sunAltAz_{sun_alt}_{sun_az}_'
    np.savetxt(cwd+path+config_name+fname+'inShadow.txt',inShadow)
    np.savetxt(cwd+path+config_name+fname+'noInShadow.txt',notInShadow)
    
def load_shadow_samples(config_name, night,lat, sun_alt, sun_az):
    cwd=os.getcwd()
    path=f'{cwd}/shadow_samples/night{night}_lat{lat}/'
    fname=f'sunAltAz_{sun_alt}_{sun_az}_'

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inShadow=np.loadtxt(path+config_name+fname+'inShadow.txt', ndmin=2)
        notInShadow=np.loadtxt(path+config_name+fname+'noInShadow.txt', ndmin=2)
    
    return inShadow, notInShadow

#solar_power
class SolarCell:
    """
    Stores location and shadowed area (fraction) each solar cell on the solar panel
    shadowed area and efficiency attributes are stored separately to allow for optimization strategies later.
    
    initialized by : x/y coordinates of bottomleftcorner of the cell, and x/y sizes of the cell
    
    methods:
    shadow_cell() : computes % of cell in shadow, given the inShadow and notInShadow samples
    """
    def __init__(self, bottomleftcorner_x, bottomleftcorner_y, size_x, size_y):
        self.center=(bottomleftcorner_x+size_x/2, bottomleftcorner_y+ size_y/2)
        self.xrange=(bottomleftcorner_x,bottomleftcorner_x+size_x)
        self.yrange=(bottomleftcorner_y,bottomleftcorner_y+size_y)
        self.area=size_x*size_y
        
        self.shadowed=0.0
    
    def shadow_cell(self, inShadow, notInShadow):
        self.inShadow_cell=[]
        self.notInShadow_cell=[]
        for px,py in inShadow:
            if self.xrange[0]<px<=self.xrange[1] and self.yrange[0]<py<=self.yrange[1]:
                self.inShadow_cell.append([px,py])
        for px,py in notInShadow:
            if self.xrange[0]<px<=self.xrange[1] and self.yrange[0]<py<=self.yrange[1]:
                self.notInShadow_cell.append([px,py])
        if (len(self.inShadow_cell)+len(self.notInShadow_cell))<5: self.shadowed=None
        else: self.shadowed=len(self.inShadow_cell)/(len(self.inShadow_cell)+len(self.notInShadow_cell))
        
class SolarCellString:
    """
    Stores the geometry of a solar cell string and its efficiency: list of cells, efficiency
    
    initialized by : list of cells, each element being an instance of SolarCell class
    
    methods:
    update_efficiency() : iterates through all cells in the string, and finds the lowest performing cell.
                          Sets the string efficiency to the lowest efficient cell
    
    """
    def __init__(self, cells_list):
        self.cells=cells_list
        self.length=len(cells_list)
        self.efficiency=0.0
    
    def update_efficiency(self):
        self.efficiency=min([(1-cell.shadowed) for cell in self.cells if cell.shadowed is not None])

class SolarCellStringConfiguration:
    """
    Stores the geometry of all the solar cell strings that form the panel
    
    initialized by : list of strings, each element being an instance of a SolarCellString class
    
    methods:
    update_shadow_cells() : iterates through each cell, in each string and updates its %shadowed given the 
                            shadow samples inShadow and notInShadow
    update_efficiency_strings() : iterates through each string, and updates its efficiency
    power_output() : calculates the instantenous power output, assuming 1kW/m^2 of solar flux
    plot() : makes a quick plot of the solar cell configuration
    """
    def __init__(self, name, strings_list):
        self.name=name
        self.strings=strings_list
    
    def update_shadow_cells(self, inShadow, notInShadow):
        for string in self.strings:
            for cell in string.cells:
                cell.shadow_cell(inShadow, notInShadow)
                
    def update_efficiency_strings(self):
        for string in self.strings:
            string.update_efficiency()
    
    def power_output(self, sun_alt):
        sun_flux=1.0*np.cos(np.pi/2-sun_alt) #kW/m^2
        output=0.0
        for string in self.strings:
            for cell in string.cells:
                output+=sun_flux*cell.area*string.efficiency
        return output
    
    def plot(self):
        for string in self.strings:
            xy=[cell.center for cell in string.cells]
            plt.plot(*zip(*xy), color=matplotlib.cm.YlOrBr(1-string.efficiency), lw=2.0)
            for cell in string.cells:
                scatter=plt.scatter(cell.center[0],cell.center[1], c=cell.shadowed if cell.shadowed is not None else 1.0,
                                      norm=matplotlib.colors.Normalize(vmin=0.0,vmax=1.0),
                                      cmap='YlOrBr', marker='s', s=300)



        plt.colorbar(scatter, ax=plt.gca(), label='% shadowed')
        plt.xlim(-0.5,0.5)
        plt.ylim(-0.5,0.5)
        plt.xlabel('East (x-axis)')
        plt.ylabel('South (y-axis)')

def cells_gridsize(panel_side, cellsize_x, cellsize_y):
    nx=panel_side//cellsize_x
    ny=panel_side//cellsize_y
    
    ncells_x=nx+1 if nx<panel_side/cellsize_x else nx
    ncells_y=ny+1 if ny<panel_side/cellsize_y else ny
    
    return int(ncells_x), int(ncells_y)