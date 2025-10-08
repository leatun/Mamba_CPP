#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "Main.h" 

namespace py = pybind11;

py::array_t<float> mamba_cpp_forward(
    py::array_t<float, py::array::c_style | py::array::forcecast> hidden_states,
    py::array_t<float, py::array::c_style | py::array::forcecast> in_proj_w,
    py::array_t<float, py::array::c_style | py::array::forcecast> in_proj_b,
    py::array_t<float, py::array::c_style | py::array::forcecast> conv1d_w,
    py::array_t<float, py::array::c_style | py::array::forcecast> conv1d_b,
    py::array_t<float, py::array::c_style | py::array::forcecast> x_proj_w,
    py::array_t<float, py::array::c_style | py::array::forcecast> dt_proj_w,
    py::array_t<float, py::array::c_style | py::array::forcecast> dt_proj_b,
    py::array_t<float, py::array::c_style | py::array::forcecast> A_log_w,
    py::array_t<float, py::array::c_style | py::array::forcecast> D_w,
    py::array_t<float, py::array::c_style | py::array::forcecast> out_proj_w,
    py::array_t<float, py::array::c_style | py::array::forcecast> out_proj_b
) {

    MambaBlockWeights* weights = new MambaBlockWeights();
    memcpy(weights->in_proj_weight, in_proj_w.data(), in_proj_w.nbytes());
    memcpy(weights->in_proj_bias,   in_proj_b.data(),   in_proj_b.nbytes());
    memcpy(weights->conv1d_weight,  conv1d_w.data(),  conv1d_w.nbytes());
    memcpy(weights->conv1d_bias, conv1d_b.data(), conv1d_b.nbytes());
    memcpy(weights->x_proj_weight,   x_proj_w.data(),   x_proj_w.nbytes());
    memcpy(weights->dt_proj_weight,  dt_proj_w.data(),  dt_proj_w.nbytes());
    memcpy(weights->dt_proj_bias, dt_proj_b.data(), dt_proj_b.nbytes());
    memcpy(weights->A_log,   A_log_w.data(),   A_log_w.nbytes());
    memcpy(weights->D,  D_w.data(),  D_w.nbytes());
    memcpy(weights->out_proj_weight, out_proj_w.data(), out_proj_w.nbytes());
    memcpy(weights->out_proj_bias,   out_proj_b.data(),   out_proj_b.nbytes());
  
    auto output = py::array_t<float, py::array::c_style>({SEQ_LEN, D_MODEL});

    auto hidden_states_ptr = static_cast<const float (*)[D_MODEL]>(hidden_states.request().ptr);
    auto output_ptr = static_cast<float (*)[D_MODEL]>(output.request().ptr);

    main_mamba_block(hidden_states_ptr, output_ptr, weights);
    delete weights;
    return output;
}

PYBIND11_MODULE(mamba_cpp_engine, m) {
    m.doc() = "Python bindings for C++ Mamba block";
    m.def("forward", &mamba_cpp_forward, "Runs the forward pass");
}