#include "Main.h"
#include <iostream>
#include "Layers.h"
#include <cmath>

void transpose_2d_D_INNER_to_SEQ_LEN(const model_dtype in[D_INNER][SEQ_LEN], model_dtype out[SEQ_LEN][D_INNER]) {
    for (int i = 0; i < D_INNER; ++i) {
        for (int j = 0; j < SEQ_LEN; ++j) {
            out[j][i] = in[i][j];
        }
    }
}

void transpose_2d_SEQ_LEN_to_D_INNER(const model_dtype in[SEQ_LEN][D_INNER], model_dtype out[D_INNER][SEQ_LEN]) {
    for (int i = 0; i < SEQ_LEN; ++i) {
        for (int j = 0; j < D_INNER; ++j) {
            out[j][i] = in[i][j];
        }
    }
}

template<size_t L, size_t D>
void transpose_ld_to_dl(const model_dtype in[L][D], model_dtype out[D][L]) {
    for (int i = 0; i < L; ++i) {
        for (int j = 0; j < D; ++j) {
            out[j][i] = in[i][j];
        }
    }
}

template<size_t L, size_t N>
void transpose_2d_SEQ_LEN_to_D_STATE(const model_dtype in[L][N], model_dtype out[N][L]) {
    for (int i = 0; i < L; ++i) {
        for (int j = 0; j < N; ++j) {
            out[j][i] = in[i][j];
        }
    }
}

void main_mamba_block(
    const model_dtype hidden_states[SEQ_LEN][D_MODEL],
    model_dtype output[SEQ_LEN][D_MODEL],
    const MambaBlockWeights* weights
) {
    static model_dtype projected_states_intermediate[SEQ_LEN][D_INNER * 2];
    static model_dtype projected_states_transposed_cpp[D_INNER * 2][SEQ_LEN];
    static model_dtype hidden_states_after_chunk_cpp[D_INNER][SEQ_LEN];
    static model_dtype gate_after_chunk_cpp[D_INNER][SEQ_LEN];
    static model_dtype x_conv[D_INNER][SEQ_LEN];
    static model_dtype hidden_states_after_conv_cpp[D_INNER][SEQ_LEN];
    static model_dtype hidden_states_rearranged_cpp[SEQ_LEN][D_INNER];
    static model_dtype ssm_parameters_cpp[SEQ_LEN][DT_RANK + D_STATE * 2];
    static model_dtype dt_raw_cpp[SEQ_LEN][DT_RANK];
    static model_dtype B_raw_cpp[SEQ_LEN][D_STATE];
    static model_dtype C_raw_cpp[SEQ_LEN][D_STATE];
    static model_dtype dt_proj_out_cpp[SEQ_LEN][D_INNER];
    static model_dtype dt_softplus_cpp[SEQ_LEN][D_INNER];
    static model_dtype discrete_time_step_cpp[D_INNER][SEQ_LEN];
    static model_dtype A_cpp[D_INNER][D_STATE];
    static model_dtype discrete_A_cpp[D_INNER][SEQ_LEN][D_STATE];
    static model_dtype discrete_B_cpp[D_INNER][SEQ_LEN][D_STATE];
    static model_dtype deltaB_u_cpp[D_INNER][SEQ_LEN][D_STATE];
    static model_dtype y_scan_raw_cpp[D_INNER][SEQ_LEN];
    static model_dtype scan_output_with_D_cpp[D_INNER][SEQ_LEN];
    static model_dtype scan_output_gated_cpp[D_INNER][SEQ_LEN];
    static model_dtype y_rearranged_cpp[SEQ_LEN][D_INNER];

    // projection
    for (int i = 0; i < SEQ_LEN; ++i) {
        linear(hidden_states[i], projected_states_intermediate[i], weights->in_proj_weight, weights->in_proj_bias, D_MODEL, D_INNER * 2);
    }
    transpose_ld_to_dl(projected_states_intermediate, projected_states_transposed_cpp);

    // split
    for (int i = 0; i < D_INNER; ++i) 
        for (int j = 0; j < SEQ_LEN; ++j) {
        hidden_states_after_chunk_cpp[i][j] = projected_states_transposed_cpp[i][j];
        gate_after_chunk_cpp[i][j] = projected_states_transposed_cpp[i + D_INNER][j];
    }

    // Tích chập, SiLU
    causal_conv1d(hidden_states_after_chunk_cpp, x_conv, (const model_dtype(*)[D_CONV])weights->conv1d_weight, weights->conv1d_bias);
    for (int i = 0; i < D_INNER; ++i) 
        for (int j = 0; j < SEQ_LEN; ++j) {
            hidden_states_after_conv_cpp[i][j] = silu(x_conv[i][j]);
    }

    // Tạo tham số động delta_t, B, C
    transpose_2d_D_INNER_to_SEQ_LEN(hidden_states_after_conv_cpp, hidden_states_rearranged_cpp);
    for (int i = 0; i < SEQ_LEN; ++i) {
        linear(hidden_states_rearranged_cpp[i], ssm_parameters_cpp[i], weights->x_proj_weight, nullptr, D_INNER, DT_RANK + D_STATE * 2);
    }

    for (int i = 0; i < SEQ_LEN; ++i) {
        for (int j = 0; j < DT_RANK; ++j) dt_raw_cpp[i][j] = ssm_parameters_cpp[i][j];
        for (int j = 0; j < D_STATE; ++j) B_raw_cpp[i][j] = ssm_parameters_cpp[i][j + DT_RANK];
        for (int j = 0; j < D_STATE; ++j) C_raw_cpp[i][j] = ssm_parameters_cpp[i][j + DT_RANK + D_STATE];
    }
    
    for (int i = 0; i < SEQ_LEN; ++i) {
        linear(dt_raw_cpp[i], dt_proj_out_cpp[i], weights->dt_proj_weight, weights->dt_proj_bias, DT_RANK, D_INNER);
    }

    for (int i = 0; i < SEQ_LEN; ++i) 
        for (int j = 0; j < D_INNER; ++j) {
            dt_softplus_cpp[i][j] = softplus(dt_proj_out_cpp[i][j]);
    }

    transpose_2d_SEQ_LEN_to_D_INNER(dt_softplus_cpp, discrete_time_step_cpp);

    // discretion
    for (int i = 0; i < D_INNER; ++i) 
        for (int j = 0; j < D_STATE; ++j) 
            A_cpp[i][j] = -std::exp(weights->A_log[i * D_STATE + j]);


    for (int d=0; d<D_INNER; ++d) {
            for (int l=0; l<SEQ_LEN; ++l) {
                for (int n=0; n<D_STATE; ++n) {
                    // discrete_A = exp(A * delta)
                    discrete_A_cpp[d][l][n] = std::exp(A_cpp[d][n] * discrete_time_step_cpp[d][l]);
                    // discrete_B = delta * B (broadcast B)
                    discrete_B_cpp[d][l][n] = discrete_time_step_cpp[d][l] * B_raw_cpp[l][n];
                    // deltaB_u = discrete_B * u
                    deltaB_u_cpp[d][l][n] = discrete_B_cpp[d][l][n] * hidden_states_after_conv_cpp[d][l];
                }
            }
        }


    // selective Scan
    scan_core(discrete_A_cpp, deltaB_u_cpp, C_raw_cpp, y_scan_raw_cpp);

    // scan_output = scan_output + (hidden_states * D)
    for (int d = 0; d < D_INNER; ++d) {
        for (int l = 0; l < SEQ_LEN; ++l) {
            scan_output_with_D_cpp[d][l] = y_scan_raw_cpp[d][l] + (hidden_states_after_conv_cpp[d][l] * weights->D[d]);
        }
    }

    // scan_output_gated = scan_output * silu(gate)
    for (int d = 0; d < D_INNER; ++d) {
        for (int l = 0; l < SEQ_LEN; ++l) {
            scan_output_gated_cpp[d][l] = scan_output_with_D_cpp[d][l] * silu(gate_after_chunk_cpp[d][l]);
        }
    }


    transpose_2d_D_INNER_to_SEQ_LEN(scan_output_gated_cpp, y_rearranged_cpp);
    
    //linear cho out_proj
    for (int i = 0; i < SEQ_LEN; ++i) {
        linear(y_rearranged_cpp[i], output[i], weights->out_proj_weight, weights->out_proj_bias, D_INNER, D_MODEL);
    }
}

