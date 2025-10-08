#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
#include <iomanip>

#include "Main.h"
#include "Layers.h"

template<typename T, size_t N>
void read_data_from_file(const std::string& filename, T (&array)[N]) {
    std::ifstream file(filename);
    if (!file.is_open()) { std::cerr << "LỖI: Không thể mở file " << filename << std::endl; exit(1); }
    for (size_t i = 0; i < N; ++i) { file >> array[i]; }
    file.close();
}

template<typename T, size_t R, size_t C>
void read_data_from_file_2d(const std::string& filename, T (&array)[R][C]) {
    std::ifstream file(filename);
    if (!file.is_open()) { std::cerr << "LỖI: Không thể mở file " << filename << std::endl; exit(1); }
    for (size_t i = 0; i < R; ++i) {
        for (size_t j = 0; j < C; ++j) {
            file >> array[i][j];
        }
    }
    file.close();
}

template<size_t R, size_t C>
void write_data_to_file_2d(const std::string& filename, const model_dtype (&array)[R][C]) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "LỖI: Không thể tạo file để ghi " << filename << std::endl;
        return;
    }
    // Đặt độ chính xác để khớp với Python (fmt='%.8e')
    file << std::scientific << std::setprecision(8);

    std::cout << "Thông tin: Đang ghi mảng shape [" << R << "][" << C << "] vào file '" << filename << "'..." << std::endl;

    for (size_t i = 0; i < R; ++i) {
        for (size_t j = 0; j < C; ++j) {
            file << array[i][j] << (j == C - 1 ? "" : " ");
        }
        file << std::endl;
    }

    file.close();
    std::cout << "Thông tin: Ghi file '" << filename << "' thành công." << std::endl;
}

// --- Hàm tiện ích để so sánh hai mảng 2D ---
template<size_t R, size_t C>
bool compare_tensors(
    const std::string& step_name, 
    const model_dtype (&cpp_tensor)[R][C], 
    const model_dtype (&golden_tensor)[R][C],
    const float atol = 1e-3, // Ngưỡng tuyệt đối: 0.001
    const float rtol = 1e-2 // Ngưỡng tương đối: 1%
) {
    std::cout << "\n--- Đang kiểm tra: " << step_name << " ---" << std::endl;
    
    int error_count = 0;
    double sum_abs_diff = 0.0;
    double sum_rel_error = 0.0;
    float max_abs_diff = 0.0f;
    float max_rel_error = 0.0f;
    float a1 = 0.0f;
    float a2 = 0.0f;
    float b1 = 0.0f;
    float b2 = 0.0f;
    int x1, x2, y1, y2;
    const int MAX_ERRORS_TO_PRINT = 10;

    for (size_t i = 0; i < R; ++i) {
        for (size_t j = 0; j < C; ++j) {
            float cpp_val = cpp_tensor[i][j];
            float py_val = golden_tensor[i][j];
            
            float abs_diff = std::abs(cpp_val - py_val);
            
            // Tính sai lệch tương đối một cách an toàn
            float rel_error = 0.0f;
            if (std::abs(py_val) > 1e-9) {
                rel_error = abs_diff / std::abs(py_val);
            }
            
            // Tích lũy các chỉ số
            sum_abs_diff += abs_diff;
            sum_rel_error += rel_error;
            if (abs_diff > max_abs_diff) {
                max_abs_diff = abs_diff;
                x1 = i;
                y1 = j;
                a1 = cpp_tensor[i][j];
                a2 = golden_tensor[i][j];
            }
            if (rel_error > max_rel_error) {
                max_rel_error = rel_error;
                x2 = i;
                y2 = j;
                b1 = cpp_tensor[i][j];
                b2 = golden_tensor[i][j];
            }

            // Kiểm tra điều kiện pass/fail (giống np.allclose)
            // Lỗi xảy ra nếu sai số lớn hơn cả ngưỡng tuyệt đối VÀ ngưỡng tương đối
            if (abs_diff > (atol + rtol * std::abs(py_val))) {
                error_count++;
                if (error_count <= MAX_ERRORS_TO_PRINT) {
                    std::cout << std::scientific << std::setprecision(6)
                              << "  Lỗi #" << error_count 
                              << " tại [" << i << "][" << j << "]: C++=" << cpp_val 
                              << ", Py=" << py_val 
                              << ", AbsDiff=" << abs_diff 
                              << ", RelDiff=" << rel_error * 100.0 << "%" << std::endl;
                }
            }
        }
    }

    int total_elements = R * C;
    double mae = sum_abs_diff / total_elements;
    double mape = (sum_rel_error / total_elements) * 100.0;

    std::cout << std::fixed << std::setprecision(8); // Chuyển về định dạng thông thường cho báo cáo
    std::cout << "  Thống kê:" << std::endl;
    std::cout << "    - Sai số tuyệt đối trung bình (MAE):   " << mae << std::endl;
    std::cout << "    - Sai số tuyệt đối lớn nhất:         " << max_abs_diff << std::endl;
    std::cout << "    -> cpp_tensor[" << x1 << "][" << y1 << "] = " << a1 << std::endl;
    std::cout << "    -> transfomer_tensor[" << x1 << "][" << y1 << "] = " << a2 << std::endl << std::endl;
    std::cout << "    - Sai số tương đối trung bình (MAPE): " << mape << "%" << std::endl;
    std::cout << "    - Sai số tương đối lớn nhất:        " << max_rel_error * 100.0 << "%" << std::endl;
    std::cout << "    -> cpp_tensor[" << x2 << "][" << y2 << "] = " << b1 << std::endl;
    std::cout << "    -> transfomer_tensor[" << x2 << "][" << y2 << "] = " << b2 << std::endl;

    if (error_count == 0) {
        std::cout << std::scientific << std::setprecision(0);
        std::cout << "  -> SUCCESS" << std::endl;
        return true;
    } else {
        std::cout << "  -> FAILURE" << std::endl;
        std::cout << "           Số phần tử không đạt: " << error_count << " / " << total_elements << std::endl;
        return false;
    }
}

template<size_t N>
void write_data_to_file(const std::string& filename, const model_dtype (&array)[N]) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "LỖI: Không thể tạo file để ghi " << filename << std::endl;
        return;
    }
    // Đặt độ chính xác để khớp với Python (fmt='%.8e')
    file << std::scientific << std::setprecision(8);

    std::cout << "Thông tin: Đang ghi mảng size [" << N << "] vào file '" << filename << "'..." << std::endl;

    // Lặp qua tất cả các phần tử
    for (size_t i = 0; i < N; ++i) {
        file << array[i] << (i == N - 1 ? "" : " "); // Thêm dấu cách, trừ phần tử cuối
    }
    // Không cần xuống dòng ở cuối để khớp với `newline=' '` của NumPy

    file.close();
    std::cout << "Thông tin: Ghi file '" << filename << "' thành công." << std::endl;
}



int main() {
    std::cout << "--- Bắt đầu Testbench Gỡ lỗi cho Mamba Block ---" << std::endl;

    MambaBlockWeights* weights_tb = new MambaBlockWeights();

    static model_dtype block_input[SEQ_LEN][D_MODEL];
    static model_dtype golden_projected_states_transposed[D_INNER * 2][SEQ_LEN];
    static model_dtype golden_hidden_states_after_chunk[D_INNER][SEQ_LEN];
    static model_dtype golden_gate_after_chunk[D_INNER][SEQ_LEN];
    static model_dtype golden_hidden_states_after_conv[D_INNER][SEQ_LEN];
    static model_dtype golden_ssm_parameters[SEQ_LEN][DT_RANK + D_STATE * 2];
    static model_dtype golden_discrete_time_step[D_INNER][SEQ_LEN];
    static model_dtype golden_B_raw[SEQ_LEN][D_STATE];
    static model_dtype golden_C_raw[SEQ_LEN][D_STATE];
    static model_dtype golden_A_raw[D_INNER][D_STATE];
    static model_dtype golden_y_scan_raw[D_INNER][SEQ_LEN];
    static model_dtype golden_final_output1[SEQ_LEN][D_MODEL];
    static model_dtype golden_final_output2[SEQ_LEN][D_MODEL];
    static model_dtype final_output_cpp[SEQ_LEN][D_MODEL];

    std::cout << "\n--- Đang tải trọng số và dữ liệu testbench ---" << std::endl;
    std::string weights_dir = "testbench/weights_130m/new/real_weights/";
    std::string data_dir = "testbench/weights_130m/new/real_txt/";

    read_data_from_file(weights_dir + "in_proj_weight.txt", weights_tb->in_proj_weight);
    read_data_from_file(weights_dir + "in_proj_bias.txt", weights_tb->in_proj_bias);
    read_data_from_file(weights_dir + "conv1d_weight.txt", weights_tb->conv1d_weight);
    read_data_from_file(weights_dir + "conv1d_bias.txt", weights_tb->conv1d_bias);
    read_data_from_file(weights_dir + "x_proj_weight.txt", weights_tb->x_proj_weight);
    read_data_from_file(weights_dir + "dt_proj_weight.txt", weights_tb->dt_proj_weight);
    read_data_from_file(weights_dir + "dt_proj_bias.txt", weights_tb->dt_proj_bias);
    read_data_from_file(weights_dir + "A_log.txt", weights_tb->A_log);
    read_data_from_file(weights_dir + "D.txt", weights_tb->D);
    read_data_from_file(weights_dir + "out_proj_weight.txt", weights_tb->out_proj_weight);
    read_data_from_file(weights_dir + "out_proj_bias.txt", weights_tb->out_proj_bias);
    
    read_data_from_file_2d(data_dir + "0_block_input.txt", block_input);
    read_data_from_file_2d(data_dir + "1_x_after_projtrans.txt", golden_projected_states_transposed);
    read_data_from_file_2d(data_dir + "2a_hidden_states_after_chunk.txt", golden_hidden_states_after_chunk);
    read_data_from_file_2d(data_dir + "2b_gate_after_chunk.txt", golden_gate_after_chunk);
    read_data_from_file_2d(data_dir + "4_hidden_states_after_conv.txt", golden_hidden_states_after_conv);
    read_data_from_file_2d(data_dir + "5a_ssm_parameters.txt", golden_ssm_parameters);
    read_data_from_file_2d(data_dir + "5b_discrete_time_step.txt", golden_discrete_time_step);
    read_data_from_file_2d(data_dir + "5c_B_raw.txt", golden_B_raw);
    read_data_from_file_2d(data_dir + "5d_C_raw.txt", golden_C_raw);
    read_data_from_file_2d(data_dir + "6a_A_raw.txt", golden_A_raw);
    read_data_from_file_2d(data_dir + "7a_scan_output_raw.txt", golden_y_scan_raw);
    read_data_from_file_2d(data_dir + "9_final_output_manual.txt", golden_final_output1);
    read_data_from_file_2d(data_dir + "9_final_output_transformer.txt", golden_final_output2);
    
    std::cout << "Tải dữ liệu thành công." << std::endl;

    //Chạy hàm mainmambablock
    main_mamba_block(block_input, final_output_cpp, weights_tb);

    if (!compare_tensors("Bước 9 (manual)", final_output_cpp, golden_final_output1));
    std::cout << "\n";
    if (!compare_tensors("Bước 9 (transformer)", final_output_cpp, golden_final_output2)) return 1;
    return 0;
}