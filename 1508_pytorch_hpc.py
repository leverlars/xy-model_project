import numpy as np
import os
import time
from natsort import natsorted
import torch
import torch.nn.functional as F

# Constants

# The interesting physics parameter :)
V_0 = 0.1
# Kinetic coefficient (timescale) should be less than or equal to 1 
Gamma = 0.5
# Determines the order of the initial state (not used)
Delta = 1
# Scalar
s = 1
# Size of lattice
N = 256
M = N
# Time step
dt = 0.001
# Amount of timesteps I want the simulation to run in units: number divided by size of dt
T = 100
# Seed to reproduce the randomization
np.random.seed(69420)
torch.manual_seed(0)
# Determines interval between images saved for plotting
step_interval = 10

# Just some testing to make sure I'm using the GPU
torch.cuda.is_available()
torch.cuda.device_count()
torch.cuda.current_device()
torch.cuda.device(0)
torch.cuda.get_device_name(0)

# Makes sure the code uses GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function that initializes a field
def create_vector_field(N, M):
    vector_field = torch.empty((N, M, 2), device=device)
    angle = torch.rand(N, M, device=device) * 2 * np.pi
    vector_field[..., 0] = s * torch.cos(angle)
    vector_field[..., 1] = s * torch.sin(angle)
    return vector_field

vector_matrix = create_vector_field(N, M)

# Function that calculates the angles of a field
def angle_from_field(vector_field):
    angle_field = torch.atan2(vector_field[..., 1] / s, vector_field[..., 0] / s)
    return angle_field

# Function that calculates the scalars of a field
def scalar_from_field(vector_field):
    scalar_field = torch.sqrt(vector_field[..., 0] ** 2 + vector_field[..., 1] ** 2)
    return scalar_field

# Function that defines the derivative/numerical scheme
def vector_field_derivative(vector_field):
    up = torch.roll(vector_field, shifts=-1, dims=0)
    down = torch.roll(vector_field, shifts=1, dims=0)
    left = torch.roll(vector_field, shifts=-1, dims=1)
    right = torch.roll(vector_field, shifts=1, dims=1)

    non_linear_term = V_0 * 2 * vector_field * (torch.sum(vector_field ** 2, dim=-1, keepdim=True) - 1)
    laplacian_term = (up + down + left + right - 4 * vector_field)

    vector_field_derived = -Gamma * (non_linear_term - 2 * laplacian_term)
    return vector_field_derived

# Computes next time step
def compute_next_step(vector_field, dt):
    derivative = vector_field_derivative(vector_field)
    vector_field_next_step = vector_field + dt * derivative
    return vector_field_next_step

# Gets angles and scalars from time step
def compute_fields(vector_field):
    angle_field = angle_from_field(vector_field)
    scalar_field = scalar_from_field(vector_field)
    return vector_field, angle_field, scalar_field

# Defines a peak detection function that works for PyTorch
def detect_peaks_torch(scalar_field, min_distance=3, threshold_rel=0.5):
    # Reshape to prevent broadcasting errors
    scalar_field = scalar_field.unsqueeze(0).unsqueeze(0)
    # Defines a field where each position contains the maximum value found in the corresponding neighbourhood of the input field
    pooled_field = F.max_pool2d(scalar_field, kernel_size=min_distance*2+1, stride=1, padding=min_distance)
    # Checks if a points is isolated from other peaks and if it's a peak itself then appends it
    mask = (scalar_field == pooled_field) & (scalar_field > threshold_rel)
    coordinates = torch.nonzero(mask.squeeze())
    
    return coordinates

def count_defects(scalar_field):
    coordinates = detect_peaks_torch(-scalar_field + 1, min_distance=3, threshold_rel=0.5)
    num_defects = len(coordinates)
    
    return num_defects

# Runs the simulation in time
def time_derivative(vector_field, dt, T):
    os.makedirs("angle_field_iterations", exist_ok=True)
    os.makedirs("scalar_field_iterations", exist_ok=True)
    defect_counts = []

    t = torch.arange(0, T, dt, device=device)

    for i in range(len(t)):
        vector_field = compute_next_step(vector_field, dt)
        vector_field_iteration, angle_field, scalar_field = compute_fields(vector_field)

        if i % step_interval == 0:
            defect_count = count_defects(scalar_field.cpu())
            defect_counts.append(defect_count)

    torch.save(defect_counts, "defect_counts.pt")

    return ["angle_field_iterations", "scalar_field_iterations"]

t0 = time.time()
vector_matrix_simulation = time_derivative(vector_matrix, dt, T)


current_defect_counts = torch.load('defect_counts.pt', weights_only=True)

torch.save('defect_counts_01.npy', current_defect_counts)

t1 = time.time()
total_time = t1 - t0
print(total_time)

# For dt=0.01, T=1000, Gamma=0.5, N=256
V_0_values = torch.Tensor([0.1, 1/3, 1, 10/3, 10, 100/3, 100])

# defect_counts_01 = torch.load('defect_counts_01.pt')
# defect_counts_03 = torch.load('defect_counts_03.pt')
# defect_counts_1 = torch.load('defect_counts_1.pt')
# defect_counts_3 = torch.load('defect_counts_3.pt')
# defect_counts_10 = torch.load('defect_counts_10.pt')
# defect_counts_33 = torch.load('defect_counts_33.pt')
# defect_counts_100 = torch.load('defect_counts_100.pt')
