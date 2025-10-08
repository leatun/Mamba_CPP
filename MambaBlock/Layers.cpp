#include "Layers.h"
#include <cmath>

// --- IMPLEMENT HÀM LINEAR ---
void linear(const model_dtype x[], model_dtype y[], 
            const model_dtype* W, const model_dtype b[],
            int in_dim, int out_dim) {
    
    // W được truyền vào dưới dạng mảng 1D, index = [hàng * chiều_rộng + cột]
    for (int i = 0; i < out_dim; ++i) {
        model_dtype sum = 0.0f;
        for (int j = 0; j < in_dim; ++j) {
            sum += x[j] * W[i * in_dim + j];
        }
        // Một số lớp (ví dụ x_proj) có thể không có bias -> b == nullptr
        y[i] = sum + (b ? b[i] : 0.0f);
    }
}

void causal_conv1d(const model_dtype x[][SEQ_LEN], model_dtype out[][SEQ_LEN],
                   const model_dtype weight[][D_CONV], const model_dtype bias[]) {
    
    const int PADDED_LEN = SEQ_LEN + D_CONV - 1;
    for (int d = 0; d < D_INNER; ++d) {
        model_dtype x_padded[PADDED_LEN] = {0.0f};
        for (int i = 0; i < SEQ_LEN; ++i) {
            x_padded[i + D_CONV - 1] = x[d][i];
        }

        for (int l = 0; l < SEQ_LEN; ++l) {
            model_dtype sum = 0.0f;
            for (int k = 0; k < D_CONV; ++k) {
                sum += x_padded[l + k] * weight[d][k];
            }
            out[d][l] = sum + bias[d];
        }
    }
}

// --- IMPLEMENT HÀM SILU ---
model_dtype silu(model_dtype x) {
    return x / (1.0f + std::exp(-x));
}

//soffmax
model_dtype softplus(model_dtype x) {
    return std::log(1.0f + std::exp(x));
}

void scan_core(
    const model_dtype discrete_A[][SEQ_LEN][D_STATE],
    const model_dtype deltaB_u[][SEQ_LEN][D_STATE],
    const model_dtype C_raw[][D_STATE],
    model_dtype scan_output_raw[][SEQ_LEN]
) {

    for (int d = 0; d < D_INNER; ++d) {
        model_dtype h[D_STATE] = {0.0f};

        for (int l = 0; l < SEQ_LEN; ++l) {
            // h_t = discrete_A_t * h_{t-1} + deltaB_u_t
            for (int n = 0; n < D_STATE; ++n) {
                h[n] = discrete_A[d][l][n] * h[n] + deltaB_u[d][l][n];
            }

            // y_t = C_t * h_t
            model_dtype y_scan = 0.0f;
            for (int n = 0; n < D_STATE; ++n) {
                y_scan += C_raw[l][n] * h[n];
            }
            scan_output_raw[d][l] = y_scan;
        }
    }
}

