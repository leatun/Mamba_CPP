A C++ implementation of the Mamba Selective State Space Model (SSM) architecture, inspired by the original 130m param pretrained model uploaded on Hugging Face.
It includes simple scripts to extract weights from pretrained checkpoints, verify block outputs against the original model, and run end-to-end evaluation with the LM Hardness benchmark.

Dependencies:  
-to be included

## 📁 Repository Structure

```
├── MambaBlock/                
│   ├── Layers.cpp             # Core implementation of Mamba block components
│   ├── Layers.h
│   ├── Main.cpp               # Example entry point for standalone testing
│   └── Main.h
│
├── pybind/
│   ├── block_binding.cpp      # Pybind11 bridge for C++ <-> Python integration
│   ├── create_hybrid_model.py # Python script to replace a Mamba block with C++ version
│   ├── hybrid_block.py        # Definition of hybrid architecture
│   └── mamba_cpp_engine.so    # Compiled shared library (generated)
│
├── weights_130m/
│   ├── new/                   # Directory for extracted weight text files
│   ├── new_weight_extract.py  # Extract weights from pretrained model
│   ├── source_search.py       # Helper for locating weight tensors
│   └── mamba_block_usage.cpp  # Compares C++ and Python outputs for verification
│
└── testbench/                 # Testing and LM evaluation setup
```

---

## Workflow Overview

1. **Extract weights**

   ```bash
   python weights_130m/new_weight_extract.py
   ```

   This extracts all necessary weight tensors into `.txt` files for the C++ model.

2. **Verify block correctness**

   ```bash
   g++ MambaBlock/Layers.cpp weights_130m/mamba_block_usage.cpp -o block_test
   ./block_test
   ```

   This compares the raw tensor output from a single C++ block against the pretrained model output (same weight, same input).

3. **Build Python bindings**

   ```bash
   cd pybind
   g++ -O3 -shared -std=c++17 -fPIC block_binding.cpp -o mamba_cpp_engine.so \
       $(python3 -m pybind11 --includes)
   ```

   Creates a Python-importable `.so` module for C++ inference.

4. **Hybrid model integration**
   Replace one of the Mamba blocks in the `130M` pretrained model with the C++ version using `create_hybrid_model.py`.

5. **Evaluation**
   Run the **LM Hardness Evaluation** library for full-model benchmarking with mixed C++ and Python layers.

---

## ⚙️ Build Notes

* Requires **g++ ≥ 9.0** and **Pybind11**.
* Tested on Ubuntu 22.04 and Python 3.10.
* Large model weights (`model.safetensors`) are **not included** in this repository.

---

## 🧩 Future Work

* 
* 
* 

---

## 📜 License

MIT License.
