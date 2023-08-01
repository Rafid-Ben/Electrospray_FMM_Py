import numpy as np
import time
from helpers import *



N=100
dt=5e-12
k=8.9875517923*1e9
vy=5000
dt=5e-12
P=2
nbpl=100

start_time = time.time()
species, pos_save = fmm_nbody(N,dt,vy,P,nbpl)
end_time = time.time()
elapsed_time = end_time - start_time
print("Condition: N= ",N, "; P= ", P, " ; nbpl= ",nbpl)
print("Time elapsed: ", elapsed_time, "seconds")


#animate_injection_3D(species,pos_save)
#FPS=500
#png_to_mp4('frames', FPS,'movie_10000.mp4')
