import numpy as np
from scipy.optimize import minimize

def error_function(coords_flattened, N, distances):
    coords = coords_flattened.reshape((N, 3))
    error = 0
    for i in range(N):
        for j in range(i+1, N):
            computed_distance = np.linalg.norm(coords[i] - coords[j])
            error += (computed_distance - distances[i, j - i - 1]) ** 2
    return error

def DGD(distances):
    N = distances.shape[0] + 1
    initial_coords = np.random.rand(N * 3)  # initial random 3D configuration
    
    result = minimize(error_function, initial_coords, args=(N, distances))
    return result.x.reshape((N, 3))
