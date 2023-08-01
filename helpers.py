import numpy as np
import exafmm.laplace as laplace
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from cmath import nan

def FMM(positions, masses, charges,P,nbpl):
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



def leapfrog_kdk(pos,vel,acc,dt,mass,charge,P,nbpl):
	"""Takes the current position, velocity, and acceleration at time t. Then, it carries out
 the leapfrog scheme kick-drift-kick. It then returns the updated position, velocity and accelration
 at time t+dt.
 	Args:
		pos (np.array of Nx3): _Position x, y, and z of N particles_
		vel (np.array of Nx3): _Velocity vx, vy, and vz of N particles_
		acc (np.array of Nx3): _Acceleration ax, ay, and az of N particles_
		dt (float): _Timestep_
		mass (np.array of N): _Mass of N particles_
        charge (np.array of N): _Charge of N particles_
        P: order of expansion of the FMM
        nbpl: max number of bodies per leaf
	Returns:
		pos (np.array of Nx3): _New position x, y, and z of N particles_
		vel (np.array of Nx3): _New velocity vx, vy, and vz of N particles_
		acc (np.array of Nx3): _New acceleration ax, ay, and az of N particles_
	"""
	# (1/2) kick 
	vel += acc * dt/2.0
	# drift
	pos += vel * dt
	# update accelerations
	acc = FMM(pos, mass, charge,P,nbpl)
	# (1/2) kick
	vel += acc * dt/2.0
 
	return pos, vel, acc


def initial_conditions(N,vy,P,nbpl):
    """Creates the initial conditions of the simulation
 
	Args:
		N (_int_): _Number of particles_ 
		vy (float, optional): _velocity in the y direction_. 
	"""
    # Generate Initial Conditions of the first injected particle 
    np.random.seed(123)    # set the random number generator seed
    #species=np.random.randint(3, size=N) #Chose one species among 3 at each timestep
    species=np.random.choice([0,0,0,0,0,0,0,0,1,2], N) #Chose one species among 3 at each timestep
    amu2kg= 1.66053906660 *1e-27 # converts amu to kg
    e2C= 1.602176634 *1e-19 # converts electron charge to Coulomb
    
    masses=np.array([111.168,309.141,197.973])*amu2kg  # define 3 different species with 3 different masses in kg 
    charges=np.array([1.,1.,0])*e2C  # define 3 different species with different charges in Coulomb
    
    mass=np.array([[masses[i] for i in list(species)]]).T  # mass of the entire set of particles
    charge=np.array([[charges[i] for i in list(species)]]).T  # charge of the entire set of particles
    
    
    mass=mass.squeeze()
    charge= charge.squeeze()
    pos=np.zeros([N,3]) # initial position of all the set of particles 
    vel= np.hstack([np.array([[0.5*vy*np.random.uniform(-1,1) for i in range(N)]]).T,\
         np.repeat([[vy]],N,axis=0),\
         np.array([[0.5*vy*np.random.uniform(-1,1) for i in range(N)]]).T]) # velocity of the enitre set of particles
    
    acc=np.zeros([N,3]) # initial acceleration of all the set of particles 
    acc[0] = FMM(pos[0:1], mass[0:1], charge[0:1],P,nbpl) # calculate initial gravitational accelerations
    return pos, vel, acc, mass, charge, species




def fmm_nbody(N,dt,vy,P,nbpl):
    """FMM computation of the N body problem. The complexity of this algorithm
    is O(N^2)
 
    Args:
		N (_int_): Number of injected particles
    	dt (_float_): _timestep_
    	vy (float, optional): _velocity in the y direction_. 
    """
    
    pos, vel, acc, mass, charge, species = initial_conditions(N,vy,P,nbpl)
	# pos_save: saves the positions of the particles at each time step
    pos_save = np.ones((N,3,N))*nan
    pos_save[0,:,0] = pos[0:1]
 
 	#vel_save: saves the velocities of the particles at each time step for computing the energy at each time step
    #vel_save = np.ones((N,3,N))*nan
    #vel_save[0,:,0] = vel[0:1]

	#species=np.random.randint(3, size=N)  # Check this one out
	# Simulation Main Loop
    for i in range(1,N):
		# Run the leapfrog scheme:
        pos[0:i],vel[0:i],acc[0:i]=leapfrog_kdk(pos[0:i],vel[0:i],acc[0:i],dt,mass[0:i],charge[0:i],P,nbpl)
    
    # save the current position and velocity of the 0 to i particles:
        pos_save[:i,:,i] = pos[0:i]
        #vel_save[:i,:,i] = vel[0:i]
		
    return species, pos_save




from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle
import matplotlib





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


