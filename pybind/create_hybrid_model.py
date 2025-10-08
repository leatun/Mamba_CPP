from transformers import AutoTokenizer, AutoModelForCausalLM
from hybrid_block import HybridMambaBlock

MODEL_NAME = "state-spaces/mamba-130m-hf"
LAYER_TO_REPLACE = 12 
HYBRID_MODEL_SAVE_PATH = "./mamba-130m-hybrid"

print(f"Đang tải model gốc: {MODEL_NAME}")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# --- In thông tin của model gốc ---
print("\n--- Thông tin mô hình gốc ---")
num_layers = len(model.backbone.layers)
print(f"  - Tổng số lớp (block): {num_layers}")

original_block = model.backbone.layers[LAYER_TO_REPLACE]
print(f"  - Block sẽ được thay thế: Layer #{LAYER_TO_REPLACE}")
print(f"  - Loại của block gốc: {original_block.__class__.__name__}")

original_mixer = original_block.mixer
print(f"  - Loại của mixer bên trong: {original_mixer.__class__.__name__}")
print("  - Cấu trúc của mixer gốc:")
print(f"    - in_proj: {original_mixer.in_proj}")
print(f"    - conv1d: {original_mixer.conv1d}")
print(f"    - out_proj: {original_mixer.out_proj}")

print(f"Sẽ thay thế layer {LAYER_TO_REPLACE}...")

original_block = model.backbone.layers[LAYER_TO_REPLACE]

hybrid_block = HybridMambaBlock(original_block)

model.backbone.layers[LAYER_TO_REPLACE] = hybrid_block
print("Thay thế thành công!")

print(f"Đang lưu mô hình lai vào '{HYBRID_MODEL_SAVE_PATH}'...")
model.save_pretrained(HYBRID_MODEL_SAVE_PATH)
tokenizer.save_pretrained(HYBRID_MODEL_SAVE_PATH)

print("Hoàn tất!")