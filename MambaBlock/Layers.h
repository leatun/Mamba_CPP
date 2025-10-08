#pragma once

// --- ĐỊNH NGHĨA CÁC HẰNG SỐ ---
const int BATCH_SIZE = 1; 
const int SEQ_LEN = 10;    
const int D_MODEL = 768;   
const int D_STATE = 16;    
const int D_CONV = 4;      
const int EXPAND = 2;     
const int D_INNER = EXPAND * D_MODEL;
const int DT_RANK = (D_MODEL + 15) / 16; 

// Kiểu dữ liệu cơ bản
typedef float model_dtype;

// --- KHAI BÁO CÁC HÀM ---
void linear(const model_dtype x[], model_dtype y[], 
            const model_dtype* W, const model_dtype b[],
            int in_dim, int out_dim);

void causal_conv1d(const model_dtype x[][SEQ_LEN], model_dtype out[][SEQ_LEN],
                   const model_dtype weight[][D_CONV], const model_dtype bias[]);

model_dtype silu(model_dtype x);

void scan_core(
    const model_dtype discrete_A[][SEQ_LEN][D_STATE],
    const model_dtype deltaB_u[][SEQ_LEN][D_STATE],
    const model_dtype C_raw[][D_STATE],
    model_dtype scan_output_raw[][SEQ_LEN]
);

model_dtype softplus(model_dtype x);
