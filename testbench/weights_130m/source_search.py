from transformers import AutoModelForCausalLM
import inspect # Thư viện để "soi" vào các đối tượng Python

MODEL_NAME = "state-spaces/mamba-130m-hf"

print(f"Đang tải model '{MODEL_NAME}' để tìm mã nguồn...")

# Tải model với trust_remote_code=True
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)

print("\nTải model thành công.")

# Lấy ra lớp của mô hình (ví dụ: MambaForCausalLM)
model_class = model.__class__
print(f"Lớp của model là: {model_class.__name__}")

# Sử dụng inspect để tìm đường dẫn đến file định nghĩa lớp này
try:
    source_file_path = inspect.getfile(model_class)
    print(f"\n>>> MÃ NGUỒN GỐC ĐƯỢC TÌM THẤY TẠI: <<<")
    print(source_file_path)
except TypeError:
    print("\nKhông thể tự động tìm thấy file. Có thể nó được định nghĩa động.")
    print("Tuy nhiên, nó thường nằm trong thư mục cache của Hugging Face.")
    # In ra thư mục cache để bạn có thể tự tìm
    from huggingface_hub import snapshot_download
    model_path = snapshot_download(MODEL_NAME)
    print(f"\nBạn có thể tìm các file mã nguồn trong thư mục:\n{model_path}")