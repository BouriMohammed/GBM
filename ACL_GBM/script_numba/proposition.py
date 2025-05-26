import numpy as np
from numba import cuda, float32
import warnings
warnings.filterwarnings("ignore")

def proposition_2(fct,interval):
    result=[]
    for i in range (interval.shape[0]):
        a = interval[i][0]
        b = interval[i][1]
        min_fct = np.min([fct(a), fct(b)])
        if a<=0 and b >=0:
            max_fct = fct(0)
        else:
            max_fct = np.max([fct(a), fct(b)])
        result.append([min_fct,max_fct])
    return np.array(result)
    

@cuda.jit
def proposition_4_kernel(omega, U, V, H, bias, T):
    i = cuda.grid(1)
    max_omega_cols = 768  # Use the maximum possible size for omega columns (300/768)
    max_U_cols = 64       # Use the maximum possible size for U columns 
    if i < omega.shape[0]:
        V_gate_i = cuda.local.array((max_omega_cols, 2), dtype=float32)
        H_gate_i = cuda.local.array((max_U_cols, 2), dtype=float32) 
        
        # Compute V_gate_i
        for j in range(omega.shape[1]):
            for k in range(V.shape[1]):
                if omega[i][j] >= 0:
                    V_gate_i[j][k] = omega[i][j] * V[j][k]
                else:
                    V_gate_i[j][k] = omega[i][j] * V[j][V.shape[1]-1-k]
        
        # Compute H_gate_i
        for l in range(U.shape[1]):
            for k in range(H.shape[1]):
                if U[i][l] >= 0:
                    H_gate_i[l][k] = U[i][l] * H[l][k]
                else:
                    H_gate_i[l][k] = U[i][l] * H[l][H.shape[1]-1-k]
        
        # Sum V_gate_i and H_gate_i
        sum_V = cuda.local.array(2, dtype=float32)
        sum_H = cuda.local.array(2, dtype=float32)
        for k in range(2):
            sum_V[k] = 0.0
            sum_H[k] = 0.0
        
        for j in range(omega.shape[1]):
            for k in range(V.shape[1]):
                sum_V[k] += V_gate_i[j][k]
                
        for l in range(U.shape[1]):
            for k in range(H.shape[1]):
                sum_H[k] += H_gate_i[l][k]
        
        for k in range(T.shape[1]):
            T[i][k] = sum_V[k] + sum_H[k] + bias[i]



def proposition_4(output_gate_0_0, output_gate_0_1, VV, HH, output_gate_1):
    # Move data to device
    omega_dev = cuda.to_device(output_gate_0_0)
    U_dev = cuda.to_device(output_gate_0_1)
    V_dev = cuda.to_device(VV)
    H_dev = cuda.to_device(HH)
    bias_dev = cuda.to_device(output_gate_1)
    T_dev = cuda.device_array((omega_dev.shape[0], 2), dtype=np.float32) 
    
    # Define grid and block sizes
    threads_per_block = 256
    blocks_per_grid = (omega_dev.shape[0] + threads_per_block - 1) // threads_per_block
    # threads_per_block = 32
    # blocks_per_grid = (64 + threads_per_block - 1) // threads_per_block

    # Launch the kernel
    proposition_4_kernel[blocks_per_grid, threads_per_block](omega_dev, U_dev, V_dev, H_dev, bias_dev, T_dev)

    # proposition_4_kernel[blocks_per_grid, threads_per_block](omega_dev, U_dev, V_dev, H_dev, bias_dev, T_dev)

    # Copy the result back to the host
    T = T_dev.copy_to_host()
    return T

 

def proposition_5(ct,V,H,C,A,B,D):
    under_V = V[:,0]
    bar_V = V[:,1]
    
    under_H = H[:,0]
    bar_H = H[:,1]
    
    under_C = C[:,0]
    bar_C = C[:,1]
    
    mid_V = (under_V + bar_V)/2
    mid_H = (under_H + bar_H)/2
    mid_C = (under_C + bar_C)/2
    
    diff_V = bar_V - under_V
    diff_H = bar_H - under_H
    diff_C = bar_C - under_C

    c = []
    Ct = ct(mid_V,mid_H,mid_C)
 
    for i in range(C.shape[0]):
        min_ci = Ct[i] - np.dot(A[i], diff_V) - np.dot(B[i], diff_H) - np.dot(D[i], diff_C)
        max_ci = Ct[i] + np.dot(A[i], diff_V) + np.dot(B[i], diff_H) + np.dot(D[i], diff_C)
        c.append([min_ci, max_ci])
    return np.array(c)

