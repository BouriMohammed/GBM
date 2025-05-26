import numpy as np
from numba import cuda, float32
import warnings
warnings.filterwarnings("ignore")

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    sig_x = sigmoid(x)
    return sig_x * (1 - sig_x)


def tanh_derivative(x):
    return 1 - np.tanh(x)**2


def interval_multiplication(array1, array2):
    # Unpack the lower and upper bounds
    x1, x2 = array1[..., 0], array1[..., 1]
    y1, y2 = array2[..., 0], array2[..., 1]
    
    # Compute all possible products
    values1 = x1 * y1
    values2 = x1 * y2
    values3 = x2 * y1
    values4 = x2 * y2
    
    # Calculate min and max for each interval using NumPy's efficient min and max functions
    interval_min = np.minimum.reduce([values1, values2, values3, values4], axis=0)
    interval_max = np.maximum.reduce([values1, values2, values3, values4], axis=0)
    
    return np.stack((interval_min, interval_max), axis=-1)
 

@cuda.jit
def float_multiplication_kernel(matrix, interval, result):
    i, j = cuda.grid(2)
    
    if i < result.shape[0] and j < result.shape[1]: 
        for k in range(interval.shape[1]):
            if matrix[i][j] >= 0:
                result[i, j][k] = matrix[i][j] * interval[i][k]
            else:
                result[i, j][k] = matrix[i][j] * interval[i][interval.shape[1]-1-k]

def float_multiplication(matrix, interval):
    # Move data to device
    matrix_dev = cuda.to_device(matrix)
    interval_dev = cuda.to_device(interval) 
    result_dev = cuda.device_array((matrix.shape[0], matrix.shape[1],2), dtype=np.float32) 

    # Define grid and block sizes
    threads_per_block = (16, 16)  # Define appropriate block size
    blocks_per_grid_x = (matrix.shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (matrix.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    # Launch the kernel
    float_multiplication_kernel[blocks_per_grid, threads_per_block](matrix_dev, interval_dev, result_dev)

    # Copy the result back to the host
    result = result_dev.copy_to_host()
    return result

