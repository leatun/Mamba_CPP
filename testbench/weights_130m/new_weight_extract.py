import torch
import torch.nn.functional as F
import numpy as np
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from einops import rearrange
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

MODEL_NAME = "state-spaces/mamba-130m-hf"

WEIGHTS_DIR = "weights_130m/new/real_weights"
TXT_DIR = "weights_130m/new/real_txt"

os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(TXT_DIR, exist_ok=True)

def save_tensor_to_txt(tensor, filename, directory, flatten=True, is_weight=False):
    """Lưu một tensor PyTorch ra file txt."""
    full_path = os.path.join(directory, filename)
    print(f"Đang lưu tensor shape {tensor.shape} vào file '{full_path}'...")
    tensor_np = tensor.detach().cpu().numpy()
    if flatten:
        tensor_np = tensor_np.flatten()
    
    if flatten and is_weight:
        np.savetxt(full_path, tensor_np, fmt='%.8e', newline=' ')
    else:
        np.savetxt(full_path, tensor_np, fmt='%.8e')

def create_empty_file(filename, directory):
    full_path = os.path.join(directory, filename)
    print(f"CẢNH BÁO: Trọng số không tồn tại. Đang tạo file trống '{full_path}'...")
    with open(full_path, 'w') as f:
        pass 

def compare_tensors(tensor_manual, tensor_golden, name="Tensor"):
    """
    So sánh hai tensor và in ra một báo cáo chi tiết.
    """
    print(f"\n--- Đang so sánh: {name} ---")
    
    tensor_manual = tensor_manual.to(torch.float32)
    tensor_golden = tensor_golden.to(torch.float32)

    abs_diff = torch.abs(tensor_manual - tensor_golden)
    mae = torch.mean(abs_diff).item()
    max_abs_diff = torch.max(abs_diff).item()

    epsilon = 1e-9
    relative_diff = abs_diff / (torch.abs(tensor_golden) + epsilon)
    mre_percent = torch.mean(relative_diff).item() * 100
    max_rel_diff_percent = torch.max(relative_diff).item() * 100

    print(f"  - Sai số tuyệt đối trung bình (MAE):   {mae:.8e}")
    print(f"  - Sai số tuyệt đối lớn nhất:         {max_abs_diff:.8e}")
    print(f"  - Sai số tương đối trung bình (MAPE): {mre_percent:.4f}%")
    print(f"  - Sai số tương đối lớn nhất:        {max_rel_diff_percent:.4f}%")

    abs_tolerance = 1e-4
    rel_tolerance = 1e-3 

    all_close = torch.all( (abs_diff < abs_tolerance) | (relative_diff < rel_tolerance) )

    if all_close:
        print("  -> KẾT QUẢ: SUCCESS (Khớp trong ngưỡng cho phép)")
        return True
    else:
        print("  -> KẾT QUẢ: FAILURE (Không khớp trong ngưỡng cho phép)")
        return False


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Sử dụng thiết bị: {device}")

    print(f"\nĐang tải model và tokenizer cho '{MODEL_NAME}'...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    print("Tải thành công!")

    embedding_layer = model.backbone.embeddings
    first_mamba_block = model.backbone.layers[0].mixer
    first_mamba_block.use_fast_path = False

    config = model.config
    D_MODEL = config.hidden_size
    D_STATE = config.state_size
    D_CONV = config.conv_kernel
    EXPAND = config.expand
    D_INNER = config.intermediate_size
    DT_RANK = (D_MODEL + 15) // 16
    BATCH_SIZE = 1

    print(f"\nĐã cô lập block đầu tiên: {first_mamba_block.__class__.__name__}")
    print(f"Các tham số được xác nhận: D_MODEL={D_MODEL}, D_STATE={D_STATE}, D_CONV={D_CONV}, EXPAND={EXPAND}, D_INNER={D_INNER}, D_STATE={D_STATE}, DT_RANK={DT_RANK}")

    print("\n--- Trích xuất và lưu trọng số của Block 0 ---")
    state_dict = first_mamba_block.state_dict()

    save_tensor_to_txt(state_dict['in_proj.weight'], "in_proj_weight.txt", WEIGHTS_DIR, is_weight=True)
    if 'in_proj.bias' in state_dict:
        save_tensor_to_txt(state_dict['in_proj.bias'], "in_proj_bias.txt", WEIGHTS_DIR, is_weight=True)
    else:
        create_empty_file("in_proj_bias.txt", WEIGHTS_DIR)

    conv_weight = state_dict['conv1d.weight'].squeeze(1)
    save_tensor_to_txt(conv_weight, "conv1d_weight.txt", WEIGHTS_DIR, flatten=False) # Lưu conv weight dưới dạng 2D
    save_tensor_to_txt(state_dict['conv1d.bias'], "conv1d_bias.txt", WEIGHTS_DIR, is_weight=True)

    save_tensor_to_txt(state_dict['x_proj.weight'], "x_proj_weight.txt", WEIGHTS_DIR, is_weight=True)
    
    save_tensor_to_txt(state_dict['dt_proj.weight'], "dt_proj_weight.txt", WEIGHTS_DIR, is_weight=True)
    save_tensor_to_txt(state_dict['dt_proj.bias'], "dt_proj_bias.txt", WEIGHTS_DIR, is_weight=True)
    
    save_tensor_to_txt(state_dict['A_log'], "A_log.txt", WEIGHTS_DIR, flatten=False) # Lưu A_log dưới dạng 2D
    save_tensor_to_txt(state_dict['D'], "D.txt", WEIGHTS_DIR, is_weight=True)

    save_tensor_to_txt(state_dict['out_proj.weight'], "out_proj_weight.txt", WEIGHTS_DIR, is_weight=True)
    if 'out_proj.bias' in state_dict:
        save_tensor_to_txt(state_dict['out_proj.bias'], "out_proj_bias.txt", WEIGHTS_DIR, is_weight=True)
    else:
        create_empty_file("out_proj_bias.txt", WEIGHTS_DIR)

    print("\n--- Tạo dữ liệu đầu vào và các giá trị trung gian cho Block 0 ---")
    
    my_sentence = "Mamba is a new state space model architecture."
    inputs = tokenizer(my_sentence, return_tensors="pt")
    input_ids = inputs['input_ids'].to(device)
    
    SEQ_LEN = input_ids.shape[1]
    print(f"\nCâu đầu vào: '{my_sentence}'")
    print(f"==> Độ dài chuỗi (SEQ_LEN): {SEQ_LEN} <==")
    print("!!! Vui lòng cập nhật hằng số SEQ_LEN trong code C++ thành giá trị này !!!")

    with torch.no_grad():
        # 0. Đầu vào của block
        block_input = embedding_layer(input_ids)
        save_tensor_to_txt(block_input.squeeze(0), "0_block_input.txt", TXT_DIR, flatten=False)
        
        # 1 & 2. Proj, Chunk, Transpose
        projected_states = first_mamba_block.in_proj(block_input).transpose(1, 2)
        save_tensor_to_txt(projected_states.squeeze(0), "1_x_after_projtrans.txt", TXT_DIR, flatten=False)

        hidden_states, gate = projected_states.chunk(2, dim=1)
        save_tensor_to_txt(hidden_states.squeeze(0), "2a_hidden_states_after_chunk.txt", TXT_DIR, flatten=False)
        save_tensor_to_txt(gate.squeeze(0), "2b_gate_after_chunk.txt", TXT_DIR, flatten=False)

        # 3 & 4. Conv1d, Slice, SiLU
        hidden_states = first_mamba_block.act(first_mamba_block.conv1d(hidden_states)[..., :SEQ_LEN])
        save_tensor_to_txt(hidden_states.squeeze(0), "4_hidden_states_after_conv.txt", TXT_DIR, flatten=False)
        
        # 5. Tạo tham số động
        ssm_parameters = first_mamba_block.x_proj(hidden_states.transpose(1, 2))
        save_tensor_to_txt(ssm_parameters.squeeze(0), "5a_ssm_parameters.txt", TXT_DIR, flatten=False)
        time_step, B, C = torch.split(ssm_parameters, [DT_RANK, D_STATE, D_STATE], dim=-1)
        
        discrete_time_step = first_mamba_block.dt_proj(time_step)
        discrete_time_step = F.softplus(discrete_time_step).transpose(1, 2) # Shape (B, D_inner, L)
        save_tensor_to_txt(discrete_time_step.squeeze(0), "5b_discrete_time_step.txt", TXT_DIR, flatten=False)
        save_tensor_to_txt(B.squeeze(0), "5c_B_raw.txt", TXT_DIR, flatten=False)
        save_tensor_to_txt(C.squeeze(0), "5d_C_raw.txt", TXT_DIR, flatten=False)
        
        # 6. Rời rạc hóa A và B
        A = -torch.exp(first_mamba_block.A_log.float()) # Shape (D_inner, D_state)
        save_tensor_to_txt(A.squeeze(0), "6a_A_raw.txt", TXT_DIR, flatten=False)
        # B và C có shape (B, L, N). Unsqueeze và broadcast
        discrete_A = torch.exp(A.unsqueeze(0).unsqueeze(2) * discrete_time_step.unsqueeze(3)) # (B, D_inner, L, N)
        discrete_B = discrete_time_step.unsqueeze(3) * B.unsqueeze(1).float() # (B, D_inner, L, N)
        deltaB_u = discrete_B * hidden_states.unsqueeze(3).float()
        
        # 7. Vòng lặp Scan thủ công
        ssm_state = torch.zeros(BATCH_SIZE, D_INNER, D_STATE, device=device, dtype=torch.float32)
        scan_outputs = []
        for i in range(SEQ_LEN):
            ssm_state = discrete_A[:, :, i, :] * ssm_state + deltaB_u[:, :, i, :]
            scan_output_i = torch.matmul(ssm_state, C[:, i, :].unsqueeze(-1))
            scan_outputs.append(scan_output_i.squeeze(-1))
        
        scan_output = torch.stack(scan_outputs, dim=-1) # Shape (B, D_inner, L)
        save_tensor_to_txt(scan_output.squeeze(0), "7a_scan_output_raw.txt", TXT_DIR, flatten=False)

        # 8. Thêm D và Gating
        scan_output = scan_output + (hidden_states * first_mamba_block.D.unsqueeze(0).unsqueeze(-1))
        save_tensor_to_txt(scan_output.squeeze(0), "7b_scan_output_with_D.txt", TXT_DIR, flatten=False)

        scan_output_gated = scan_output * first_mamba_block.act(gate)
        save_tensor_to_txt(scan_output_gated.squeeze(0), "7c_scan_output_gated.txt", TXT_DIR, flatten=False)

        # 9. out_proj
        final_output_manual = first_mamba_block.out_proj(scan_output_gated.transpose(1, 2))
        save_tensor_to_txt(final_output_manual.squeeze(0), "9_final_output_manual.txt", TXT_DIR, flatten=False)

        # --- KIỂM TRA TÍNH TƯƠNG ĐƯƠNG ---
        print("\n--- Đang kiểm tra tính tương đương ---")
        golden_output = first_mamba_block(block_input)
        save_tensor_to_txt(golden_output.squeeze(0), "9_final_output_transformer.txt", TXT_DIR, flatten=False)
        test_passed = compare_tensors(final_output_manual, golden_output, name="Kết quả cuối cùng của Block")

    print("\nĐã tạo tất cả các file cho testbench thành công!")