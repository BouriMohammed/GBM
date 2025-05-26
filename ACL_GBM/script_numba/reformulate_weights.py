def weights(W, U, BW, BU, hidden_size: int, cell: str):
    index_map = {'i': 0, 'f': 1, 'g': 2, 'o': 3}
    start_index = index_map[cell] * hidden_size
    end_index = start_index + hidden_size

    w_i = W[start_index:end_index]
    u_i = U[start_index:end_index]
    bw_i = BW[start_index:end_index]
    bu_i = BU[start_index:end_index]

    return (w_i, u_i), bw_i + bu_i


def gates(weight_ih, weight_hh, bias_ih, bias_hh, hidden_size):
    input_gate = weights(weight_ih, weight_hh, bias_ih,
                         bias_hh, hidden_size, 'i')   # (w_i, u_i), bw_i+bu_i
    forget_gate = weights(weight_ih, weight_hh, bias_ih,
                          bias_hh, hidden_size, 'f')  # (w_f, u_f), bw_f+bu_f
    cell_gate = weights(weight_ih, weight_hh, bias_ih, bias_hh,
                        hidden_size, 'g')    # (w_g, u_g), bw_g+bu_g
    output_gate = weights(weight_ih, weight_hh, bias_ih,
                          bias_hh, hidden_size, 'o')  # (w_o, u_o), bw_o+bu_o
    return (input_gate, forget_gate, cell_gate, output_gate)
