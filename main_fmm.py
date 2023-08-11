import numpy as np
import time
from helpers_FMM import *
import pandas as pd
import faulthandler
faulthandler.enable()





info = np.loadtxt('E_field/info.txt')
convert_dist=info[1] 
convert_E=info[2]
convert_vel=info[3]

prob = np.loadtxt('E_field/prob.txt')
ri = np.loadtxt('E_field/r.txt')*convert_dist  # r initial condition
zi = np.loadtxt('E_field/z.txt')*convert_dist  # z initial condition
vri = np.loadtxt('E_field/v_r.txt')*convert_vel# vr initial condition
vzi = np.loadtxt('E_field/v_z.txt')*convert_vel # vz initial condition


# Load the csv file
df = pd.read_csv('E_field/Fields.csv')
# Convert the DataFrame to a numpy array
Field = df.values
r=Field[:,0]*convert_dist
z=Field[:,1]*convert_dist
Er=Field[:,2]*convert_E
Ez=Field[:,3]*convert_E





N=1000
dt=5e-12
Pneut=0
Pmono=50
Pdim=10
Ptrim=50
P=2
nbpl=100

interp=triangulation (r,z,Er,Ez)


start_time = time.time()
species, pos_save,IC = FMM_nbody(dt,N,prob,ri,zi,vri,vzi,Pneut,Pmono,Pdim,Ptrim,P,nbpl,interp)
end_time = time.time()
elapsed_time = end_time - start_time
print("Condition: N= ",N, "; P= ", P, " ; nbpl= ",nbpl)
print("Time elapsed: ", elapsed_time, "seconds")


#animate_injection_2D(species,pos_save)
#FPS=500
#png_to_mp4('frames', FPS,'movie_10000.mp4')
