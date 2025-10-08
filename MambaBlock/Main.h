#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
#include <iomanip>
#include "Layers.h"

struct MambaBlockWeights {
    model_dtype in_proj_weight[D_INNER * 2 * D_MODEL];
    model_dtype in_proj_bias[D_INNER * 2];
    
    model_dtype conv1d_weight[D_INNER * D_CONV];
    model_dtype conv1d_bias[D_INNER];
    
    model_dtype x_proj_weight[ (DT_RANK + D_STATE * 2) * D_INNER ];
    
    model_dtype dt_proj_weight[D_INNER * DT_RANK];
    model_dtype dt_proj_bias[D_INNER];
    
    model_dtype A_log[D_INNER * D_STATE];
    model_dtype D[D_INNER];
    
    model_dtype out_proj_weight[D_MODEL * D_INNER];
    model_dtype out_proj_bias[D_MODEL];
};


void main_mamba_block(
    const model_dtype hidden_states[SEQ_LEN][D_MODEL],
    model_dtype output[SEQ_LEN][D_MODEL],
    const MambaBlockWeights* weights
);
