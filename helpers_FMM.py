import numpy as np
import exafmm.laplace as laplace
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from cmath import nan
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator



def FMM_acc_poisson(positions, masses, charges,P,nbpl):
    """
    Compute the accelerations of a collection of particles interacting electrostatically.

    Parameters:
    - positions (List[Tuple[float, float, float]]): A list of tuples, where each tuple represents the x, y, z coordinates of a particle's position.  
    - masses (List[float]): A list of mass values for each particle. 
    - charges (List[float]): A list of charge values for each particle.
    - P: order of expansion of the FMM
    - nbpl: max number of bodies per leaf
    Returns:
    - accelerations: A numpy array of shape (n, 3) representing the acceleration of each particle in the x, y, z directions.
    """
    
    # Ensure that the inputs are numpy arrays
    #positions = np.array(positions)
    #masses = np.array(masses)
    #charges = np.array(charges)
    

    # Number of particles
    n = len(positions)

    # create a list of source instances
    sources = laplace.init_sources(positions, charges)
    # create a list of target instances
    targets = laplace.init_targets(positions)

    # create a LaplaceFmm instance
    fmm = laplace.LaplaceFmm(p=P, ncrit=nbpl, filename="test_file.dat")

    # setup the tree
    tree = laplace.setup(sources, targets, fmm)

    # evaluate potential and its gradient
    trg_values = laplace.evaluate(tree, fmm)

    # compute forces on each particle
    k=8.9875517923*1e9 # Coulomb constant
    forces = -k*charges[:, np.newaxis] * trg_values[:, 1:]  # Multiply charges by electric field

    # compute accelerations of each particle
    accelerations = forces / masses[:, np.newaxis]
     
    return accelerations










def IC_conditions (n,prob,ri,zi,vri,vzi,Pneut,Pmono,Pdim,Ptrim):
    
    def injection_conditions(n,prob,ri,zi,vri,vzi):
        probabilities = prob / prob.sum()
        np.random.seed(123)    # set the random number generator seed
        indices = np.random.choice(np.arange(len(probabilities)), size=n, p=probabilities) # Indices are distributes based on prob
        theta_i=np.random.uniform(0, 2*np.pi, np.size(indices)) # The angle is uniformly distributed
        x=ri[indices]*np.cos(theta_i)
        y=ri[indices]*np.sin(theta_i)
        z=zi[indices]
        vx=vri[indices]*np.cos(theta_i)
        vy=vri[indices]*np.sin(theta_i)
        vz=vzi[indices] 
        init_posvel=np.column_stack((x,y,z,vx,vy,vz))
        return init_posvel
    
    
    def species(n,Pneut,Pmono,Pdim,Ptrim):
    
        # Neutrals-->0 ; Monomers-->1, dimers -->2 ; Trimers -->3
        particles = np.array([0, 1, 2, 3]) 
        probabilities = np.array([Pneut,Pmono, Pdim, Ptrim])  
        # Normalizing probabilities (making sure they add up to 1)
        probabilities = probabilities / probabilities.sum()
        # Generate the array
        particle_types = np.random.choice(particles, size=n, p=probabilities)
        
        return particle_types
    
    

    amu2kg= 1.66053906660 *1e-27 # converts amu to kg
    e2C= 1.602176634 *1e-19 # converts electron charge to Coulomb
    init_posvel=injection_conditions(n,prob,ri,zi,vri,vzi)
    particle_types=species(n,Pneut,Pmono,Pdim,Ptrim)
    charges=np.sign(particle_types)*e2C
    mass_list=np.array([197.973,111.168,309.141,507.114])*amu2kg  # mass in kg: neutral, monomer, dimer, trimer 
    masses=np.array([[mass_list[i] for i in list(particle_types)]]).T  # mass of the entire set of particles
    IC=np.column_stack((init_posvel,particle_types,masses,charges))
    return IC






# Add Background Electric Field (Laplace field)

def triangulation (r,z,Er,Ez):
    points = np.column_stack((r, z))
    tri = Delaunay(points)
    E_array= np.vstack((Er.ravel(), Ez.ravel())).T
    interp=LinearNDInterpolator(tri, E_array, fill_value=np.nan, rescale=False)
    return interp


def interp_lin_delaunay(interp,request_pts):
    return interp(request_pts)



def compute_acc_laplace (interp,pos,mass,charge):
    x = pos[:,0].copy()
    y = pos[:,1].copy()
    z = pos[:,2].copy()
    r = np.sqrt(x**2 + y**2) # convert cartesian to cylindrical
    request_pts=np.vstack((r,z)).T
    E_array=interp_lin_delaunay(interp,request_pts) # nx2 array of Er and Ez
    F_cyl=charge.reshape(-1, 1)*E_array # nx2 array of Fr and Fz
    a_lap_cyl=F_cyl/mass.reshape(-1, 1)  # nx2 array of ar and az
    #convert cylindrical coordinates to cartesian coordinates
    a_lap_cart = np.zeros((a_lap_cyl.shape[0], 3)) # define an array for the acceleration in the cartesian coordinates ax, ay, az
    theta =np.arctan2(y, x) # Angle in the cylindrical coordinates formed by the point (x,y)
    a_lap_cart[:,0]=a_lap_cyl[:,0]*np.cos(theta) # a_cart(x) =a_cyl(r)*cos(theta)
    a_lap_cart[:,1]=a_lap_cyl[:,0]*np.sin(theta) # a_cart(x) =a_cyl(r)*cos(theta)
    a_lap_cart[:,2]=a_lap_cyl[:,1] # a_cart(z) =a_cyl(z)
    return a_lap_cart








def leapfrog_kdk(pos, vel, acc, dt, mass, charge, P, nbpl, interp, current_step):
    """
    Modified leapfrog scheme for particle motion based on z position.
    
    Args:
        pos (np.array of Nx3): Position x, y, and z of N particles.
        vel (np.array of Nx3): Velocity vx, vy, and vz of N particles.
        acc (np.array of Nx3): Acceleration ax, ay, and az of N particles.
        dt (float): Timestep.
        mass (np.array of N): Mass of N particles.
        k (float, optional): Coulomb constant.
        softening (float): Softening length.
        interp: Interpolation function (not detailed in given code).
        current_step (int): Current timestep.

    Returns:
        pos (np.array of Nx3): New position x, y, and z of N particles.
        vel (np.array of Nx3): New velocity vx, vy, and vz of N particles.
        acc (np.array of Nx3): New acceleration ax, ay, and az of N particles.
    """
    
    # Mask for particles with z <= 5e-6
    mask1 = pos[:,2] <= 5e-6
    # Mask for particles with 5e-6 < z <= 250e-6
    mask2 = (pos[:,2] > 5e-6) & (pos[:,2] <= 250e-6)
    # Mask for particles with z > 250e-6
    mask3 = pos[:,2] > 250e-6
    
    # Mask for region 1 and 2
    mask12=mask1 | mask2
    
    
    # (1/2) kick for particles with z <= 5e-6 or 5e-6 < z <= 250e-6
    vel[mask12] += acc[mask12] * dt/2.0
    
    # Drift for all particles
    pos += vel * dt
    
    # Update accelerations for particles with z <= 5e-6
    acc_poisson1 = FMM_acc_poisson(pos[mask1], mass[mask1], charge[mask1],P,nbpl)
    acc_laplace1 = compute_acc_laplace(interp, pos[mask1], mass[mask1], charge[mask1])
    acc[mask1] = acc_poisson1 + acc_laplace1
    
    # Update accelerations for particles with 5e-6 < z <= 250e-6
    if np.any(mask2):
        if current_step % 10 == 0:
            acc_poisson2 = FMM_acc_poisson(pos[mask2], mass[mask2], charge[mask2], P,nbpl)
        else:
            acc_poisson2 = np.zeros_like(pos[mask2])
    
        acc_laplace2 = compute_acc_laplace(interp, pos[mask2], mass[mask2], charge[mask2])
        acc[mask2] = acc_poisson2 + acc_laplace2
    
    # Acceleration is null for particles with z > 250e-6
    acc[mask3] = 0.0
    
    # (1/2) kick for particles with z <= 5e-6 or 5e-6 < z <= 250e-6
    vel[mask12] += acc[mask12] * dt/2.0

    return pos, vel, acc





def FMM_nbody(dt,N,prob,ri,zi,vri,vzi,Pneut,Pmono,Pdim,Ptrim,P,nbpl,interp):
    """Direct Force computation of the N body problem. The complexity of this algorithm
    is O(N^2)
 
    Args:
        N (_int_): Number of injected particles
        dt (_float_): _timestep_
        softening (float, optional): _softening parameter_. Defaults to 0.01.
        k (float, optional): _Coulomb constant_. Defaults to 8.9875517923*1e9.
        vy (float, optional): _velocity in the y direction_. Defaults to 50.0.
    """
    IC=IC_conditions (N,prob,ri,zi,vri,vzi,Pneut,Pmono,Pdim,Ptrim)
    IC_copy=np.copy(IC)
    pos=IC[:,0:3]
    vel=IC[:,3:6]
    species=IC[:,6]
    mass=IC[:,7]
    charge=IC[:,8]
    
    mass=mass.reshape(-1, 1)
    charge=charge.reshape(-1, 1)
    
    acc=np.zeros([N,3]) # initial acceleration of all the set of particles
     
    # pos_save: saves the positions of the particles at each time step
    pos_save = np.ones((N,3,N))*np.nan
    pos_save[0,:,0] = pos[0:1]
 
     #vel_save: saves the velocities of the particles at each time step for computing the energy at each time step
    #vel_save = np.ones((N,3,N))*nan
    #vel_save[0,:,0] = vel[0:1]

    # Simulation Main Loop
    current_step=0
    for i in range(1,N):
        # Run the leapfrog scheme:
        pos[0:i],vel[0:i],acc[0:i]=leapfrog_kdk(pos[0:i],vel[0:i],acc[0:i],dt,mass[0:i].ravel(),charge[0:i].ravel(), P, nbpl,interp,current_step)
        # save the current position and velocity of the 0 to i particles:
        pos_save[:i,:,i] = pos[0:i]
        #vel_save[:i,:,i] = vel[0:i]
        current_step += 1
        
    return species, pos_save , IC_copy










#####################



def animate_injection_2D(species,pos_save):    
    """Takes the list of species and the position of each particle at each timestep and creates
 an animation in real time.

    Args:
        species (np.array of Nx1): _Contains the type of the species of the N particles_
        pos_save (np.array of Nx3xN): _Contains the position of the particles at each time step_
    """

    N=len(species)
    colors=["forestgreen","navy","fuchsia"]
    col_list=["forestgreen"]
    col_list=[colors [int(i)] for i in list(species[:-1])]
    fig = plt.figure(figsize=(4,5), dpi=80)
    grid = plt.GridSpec(1, 1, wspace=0.0, hspace=0.3)
    ax1 = plt.subplot(grid[0,0])
 
    xmax=np.nanmax(pos_save[:,0,-1])
    ymax=np.nanmax(pos_save[:,1,-1])
    for i in range(1,N):
        
        plt.sca(ax1)
        plt.cla()
        plt.title('Number of injected particles: %i' %i, fontsize=10)
        xx = pos_save[:i,0,i]
        yy = pos_save[:i,1,i]
        plt.scatter(xx,yy,s=5,color=col_list[:i])
        ax1.set(xlim=(-xmax, xmax), ylim=(-1e-7, ymax))
        ax1.set_aspect('equal', 'box')
        plt.pause(1e-5)
        ax1.set_xlabel('X axis')
        ax1.set_ylabel('Y axis')
        
    return 0





import matplotlib.patches as mpatches
import glob
from PIL import Image
import imageio
import os


def animate_injection_3D(species,pos_save):    
    """Takes the list of species and the position of each particle at each timestep and creates
 an animation in real time.

    Args:
        species (np.array of Nx1): _Contains the type of the species of the N particles_
        pos_save (np.array of Nx3xN): _Contains the position of the particles at each time step_
    """

    N=len(species)
    colors=["forestgreen","navy","fuchsia"]
    col_list=[colors [int(i)] for i in list(species[:-1])]

    mono_patch = mpatches.Patch(color='forestgreen', label='Monomer')
    dim_patch = mpatches.Patch(color='navy', label='Dimer')
    neut_patch = mpatches.Patch(color='fuchsia', label='Neutral')

    xmax=np.nanmax(pos_save[:,0,-1])
    ymax=np.nanmax(pos_save[:,1,-1])
    zmax=np.nanmax(pos_save[:,1,-1])
 

    
    fig = plt.figure(figsize=(10,10), dpi=80)
    grid = plt.GridSpec(1, 1, wspace=0.0, hspace=0.3)
    ax = plt.axes(projection ='3d')

    for i in range(1,N):
        ax.cla()
        xx = pos_save[:i,0,i]
        yy = pos_save[:i,1,i]
        zz = pos_save[:i,2,i]
        ax.scatter(xx,yy,zz,color=col_list[:i])
        ax.set(xlim=(-ymax, ymax), ylim=(0, ymax),zlim=(-ymax, ymax))
        plt.title('Number of injected particles: %i' %i, fontsize=16)
        plt.legend(handles=[mono_patch, dim_patch,neut_patch])
        ax.set_aspect('auto', 'box')
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        #ax.set_xticks([-200,-100,0,100,150,200])
        #ax.set_yticks([0,100,200,300,400,500,600])
        ax.view_init(elev=30., azim=35)
        plt.savefig(f'frames/frame_{i:04}.png',dpi=240)  # Save each frame as a .png image
      
   
    #img, *imgs = [Image.open(f) for f in sorted(glob.glob("frames/*.png"))]
    #img.save(fp="particles_animation.gif", format='GIF', append_images=imgs, save_all=True, duration=100, loop=0)
    
    return 0






def png_to_mp4(folder, FPS, output_file):
    file_names = sorted((fn for fn in os.listdir(folder) if fn.endswith('.png')))
    
    with imageio.get_writer(output_file, mode='I', fps=FPS) as writer:
        for filename in file_names:
            image = imageio.imread(os.path.join(folder, filename))
            writer.append_data(image)





        
def animate_particles_Version0(positions_list, num_steps):
    # Prepare the figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scat = ax.scatter([], [], [])

    # Set the limits of the plot
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)

    # Prepare the animation
    def update(frame):
        scat._offsets3d = positions_list[frame].T
        return scat,

    ani = FuncAnimation(fig, update, frames=range(num_steps), blit=True)

    # Save the animation to a file
    ani.save('particles_animation.mp4', writer='ffmpeg')


