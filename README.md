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
notice: all the shell commands arent correctly configured, and are only here for instruction purposes.
we hope u dont have a hard time figuring out all this.

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
   Replace one of the Mamba blocks in the `130M` pretrained model with the C++ version and create the hybrid model using `create_hybrid_model.py`.

5. **Evaluation**  
   Run the **LM Hardness Evaluation** library for full-model benchmarking with mixed C++ and Python layers using these commands:  
     
   - Hybrid model:
   ```bash
   lm_eval --model hf --model_args pretrained=./mamba-130m-hybrid,trust_remote_code=True --tasks lambada_openai,hellaswag,arc_easy,winogrande --device cuda:0 --batch_size auto --output_path ./eval_results/hybrid_130m
   ```
   - Original model:
   ```bash
   lm_eval --model hf --model_args pretrained=state-spaces/mamba-130m-hf,trust_remote_code=True --tasks lambada_openai,hellaswag,arc_easy,winogrande --device cuda:0 --batch_size auto --output_path ./eval_results/mamba-130m-h
   ```
   This will result in two .json files with the scores for every chosen task from the **LM Hardness Evaluation** library. You can conpare it yourself to see the differences. Our checkpoints can be seen in the **eval_results** folder.
   
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

None.
CEUIT
