#!/usr/bin/env python

import numpy as np
from numba import njit, prange
import os
import time
from skimage.feature import peak_local_max
from natsort import natsorted

# Constants

# The interesting physics parameter :)
V_0 = 1/3
# Kinetic coefficient (timescale) should be less than or equal to 1 
Gamma = 0.5
# Determines the order of the initial state (not used)
Delta = 1
# Scalar
s = 1
# Size of lattice
N = 256
M = N
# Timestep
dt = 0.01
# Amount of timesteps I want the simulation to run in units: number divided by size of dt
T = 1000
# Seed to reproduce the randomization
np.random.seed(69420)
# Determines interval between images saved for plotting
step_interval = 10


# Function that initializes a field
# N and M can't currently be different because angle_field and scalar_field only uses M
@njit
def create_vector_field(N, M):
    vector_field = np.empty((N, M, 2))
    for i in range(N):
        for j in range(M):
            angle = np.random.uniform(0, 2 * np.pi)
            vector_field[i, j, 0] = s * np.cos(angle)
            vector_field[i, j, 1] = s * np.sin(angle)
    return vector_field

vector_matrix = create_vector_field(N, M)

# Function that calulates the angles of a field
@njit
def angle_from_field(vector_field):
    angle_field = np.empty((N, M))
    for i in range(N):
        for j in range(M):
            angle_field[i, j] = np.arctan2(vector_field[i, j, 1]/s, vector_field[i, j, 0]/s)
    return angle_field

# Function that calulates the scalars of a field
@njit
def scalar_from_field(vector_field):
    scalar_field = np.empty((N, M))
    for i in range(N):
        for j in range(M):
            scalar_field[i, j] = np.sqrt(vector_field[i, j, 0] ** 2 + vector_field[i, j, 1] ** 2)
    return scalar_field

# Function that defines the derivative/numerical scheme
@njit(parallel=True)
def vector_field_derivative(vector_field):
    vector_field_derived = np.zeros_like(vector_field)
    for i in prange(N):
        for j in prange(M):
            up = (i - 1) % N
            down = (i + 1) % N
            left = (j - 1) % M
            right = (j + 1) % M
            
            non_linear_term = V_0 * 2 * vector_field[i, j] * ((vector_field[i, j, 0] ** 2 + vector_field[i, j, 1] ** 2) - 1)
            laplacian_term = ((vector_field[up, j] + vector_field[down, j] + vector_field[i, left] + vector_field[i, right]) - 4 * vector_field[i, j]) / 1
            
            vector_field_derived[i, j] = -Gamma * (non_linear_term - 2 * laplacian_term)
    
    return vector_field_derived


# Computes next time step
@njit
def compute_next_step(vector_field, dt):
    derivative = vector_field_derivative(vector_field.copy())
    vector_field += dt * derivative
    return vector_field

# Gets angles and scalars from time step
@njit
def compute_fields(vector_field):
    angle_field = angle_from_field(vector_field.copy())
    scalar_field = scalar_from_field(vector_field.copy())
    return vector_field.copy(), angle_field, scalar_field



# Runs the simulation in time and counts defects
defect_counts = []

def time_derivative(vector_field, dt, T):
    
    t = np.arange(0, T, dt)
    
    for i in range(len(t)):
        vector_field = compute_next_step(vector_field.copy(), dt)
        scalar_field = compute_fields(vector_field.copy())[2]
        
        coordinates = peak_local_max(-scalar_field + 1, min_distance=3, threshold_rel=0.5, exclude_border=False)
        num_defects = len(coordinates)
        defect_counts.append(num_defects)

t0 = time.time()
vector_matrix_simulation = time_derivative(vector_matrix, dt, T)
np.save('defect_counts_03.npy', np.array(defect_counts))
t1 = time.time()
total_time = t1-t0
print(total_time)


#np.save('V_0_values.npy', np.array([0.1, 1/3, 1, 10/3, 10, 100/3, 100]))



#For dt=0.01, T=1000, Gamma=0.5, N=256
#V_0_values = np.load('V_0_values.npy')
#print(V_0_values)
# defect_counts_01 = np.load('defect_counts_01.npy')
# defect_counts_03 = np.load('defect_counts_03.npy')
# defect_counts_1 = np.load('defect_counts_1.npy')
# defect_counts_3 = np.load('defect_counts_3.npy')
# defect_counts_10 = np.load('defect_counts_10.npy')
# defect_counts_33 = np.load('defect_counts_33.npy')
# defect_counts_100 = np.load('defect_counts_100.npy')
