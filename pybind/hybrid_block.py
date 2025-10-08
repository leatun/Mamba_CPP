import torch
import torch.nn as nn
import numpy as np

import mamba_cpp_engine 

class HybridMambaBlock(nn.Module):
    def __init__(self, original_pytorch_block):
        super().__init__()
        
        self.norm = original_pytorch_block.norm
        
        mixer_state_dict = original_pytorch_block.mixer.state_dict()
        
        self.weights = {}
        for name, param in mixer_state_dict.items():
            self.weights[name] = param.detach().cpu().numpy()
            
        if 'in_proj.bias' not in self.weights:
            self.weights['in_proj.bias'] = np.array([], dtype=np.float32)
        if 'out_proj.bias' not in self.weights:
            self.weights['out_proj.bias'] = np.array([], dtype=np.float32)

    def forward(self, hidden_states: torch.Tensor, cache_params=None, cache_position=None, attention_mask=None):
        
        residual = hidden_states
        
        hidden_states_norm = self.norm(hidden_states)
        
        input_np = hidden_states_norm.detach().cpu().numpy()
        
        output_np = mamba_cpp_engine.forward(
            input_np,
            self.weights['in_proj.weight'],
            self.weights['in_proj.bias'],
            self.weights['conv1d.weight'].squeeze(1),
            self.weights['conv1d.bias'],
            self.weights['x_proj.weight'],
            self.weights['dt_proj.weight'],
            self.weights['dt_proj.bias'],
            self.weights['A_log'],
            self.weights['D'],
            self.weights['out_proj.weight'],
            self.weights['out_proj.bias']
        )
        
        mixer_output = torch.from_numpy(output_np).to(hidden_states.device)
        
        output = residual + mixer_output
        
        return output