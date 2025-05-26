import numpy as np
from script_numba.reformulate_weights import gates
from script_numba.proposition import proposition_2, proposition_4, proposition_5
from script_numba.functions import sigmoid, sigmoid_derivative, tanh_derivative, interval_multiplication, float_multiplication
import warnings
warnings.filterwarnings("ignore")


def growth_bounds_matrix(weights, VV, HH, CC, hidden_size):
    
    weight_ih, weight_hh, bias_ih, bias_hh = weights
    input_gate, forget_gate, cell_gate, output_gate = gates(weight_ih, weight_hh, bias_ih, bias_hh, hidden_size)

    def C2_t(V,H,C):
        
        f = sigmoid(np.dot(forget_gate[0][0],V) + np.dot(forget_gate[0][1],H) + forget_gate[1])
        I = sigmoid(np.dot(input_gate[0][0],V) + np.dot(input_gate[0][1],H) + input_gate[1])
        g = np.tanh(np.dot(cell_gate[0][0],V) + np.dot(cell_gate[0][1],H) + cell_gate[1])
        
        return np.multiply(f,C) + np.multiply(I,g)

    def GBM(BM):
        return np.max(np.abs(BM), axis=-1)

    T_o = proposition_4(output_gate[0][0], output_gate[0][1], VV, HH, output_gate[1])
    T_f = proposition_4(forget_gate[0][0], forget_gate[0][1], VV, HH, forget_gate[1])
    T_I = proposition_4(input_gate[0][0], input_gate[0][1], VV, HH, input_gate[1])
    T_g = proposition_4(cell_gate[0][0], cell_gate[0][1], VV, HH, cell_gate[1])

    sigma_Tf = sigmoid(T_f)
    sigma_TI = sigmoid(T_I)
    sigma_To = sigmoid(T_o)
    tanh_Tg = np.tanh(T_g)

################################### JACOBIAN OF C ###################################

    def jaccobian_C(variable):
        if variable == 'v':
            var = 0
        elif variable == 'h':
            var = 1
        elif variable == 'c':
            T1 = sigma_Tf.copy()
            T2 = T1[np.newaxis,:,:]
            return np.repeat(T2, hidden_size, axis=0)
        else:
            return -1
        
        T11 = interval_multiplication(proposition_2(sigmoid_derivative, T_f), CC)
        T1 = float_multiplication(forget_gate[0][var], T11)

        T21 = interval_multiplication(proposition_2(sigmoid_derivative, T_I), tanh_Tg)
        T2 = float_multiplication(input_gate[0][var], T21)

        T31 = interval_multiplication(proposition_2(tanh_derivative, T_g), sigma_TI)
        T3 = float_multiplication(cell_gate[0][var], T31)
        return T1 + T2 + T3

    C_v = jaccobian_C('v')
    C_h = jaccobian_C('h')
    C_c = jaccobian_C('c')

    Ac = GBM(C_v)
    Bc = GBM(C_h)
    Dc = GBM(C_c)

################################### JACOBIAN OF h ###################################
    
    def jaccobian_H(variable):
        if variable == 'v':
            var = 0
            var2 = C_v
        elif variable == 'h':
            var = 1
            var2 = C_h
        elif variable == 'c':
            T1 = interval_multiplication(interval_multiplication(sigma_To, sigma_Tf), proposition_2(tanh_derivative,proposition_5(C2_t,VV,HH,CC,Ac,Bc,Dc)))
            T2 = T1[np.newaxis,:,:]
            return np.repeat(T2, hidden_size, axis=0)
        else:
            return -1
        
        c = proposition_5(C2_t,VV,HH,CC,Ac,Bc,Dc)
        T11 = interval_multiplication(proposition_2(sigmoid_derivative, T_o), np.tanh(c))
        T1 = float_multiplication(output_gate[0][var], T11)

        T21 = interval_multiplication(proposition_2(tanh_derivative, c), sigma_To)
        T22 = T21.copy()
        T23 = T22[:,np.newaxis,:]
        T2 = interval_multiplication(var2, np.repeat(T23, output_gate[0][var][1].shape, axis=1))
        
        return T1 + T2

    H_v = jaccobian_H('v')
    H_h = jaccobian_H('h')
    H_c = jaccobian_H('c')

    A = GBM(H_v)
    B = GBM(H_h)
    D = GBM(H_c)
    
    return A,B,D
    # return H_v, H_h, H_c

