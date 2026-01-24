import torch
import os

# Đường dẫn đến file bạn vừa lưu
save_path = 'results/run_0_1/client_layer_1_epoch_2.pt' # Sửa lại đường dẫn nếu cần

try:
    # Load state_dict lên (map_location='cpu' để tránh lỗi nếu bạn không có GPU lúc check)
    state_dict = torch.load(save_path, map_location='cpu', weights_only=True)

    print(f"{'TÊN LỚP (KEY)':<30} | {'KÍCH THƯỚC (SHAPE)':<25}")
    print("-" * 80)

    # Duyệt qua từng item để in
    for param_tensor in state_dict:
        # In tên layer và kích thước tensor của nó
        print(f"{param_tensor:<30} | {str(state_dict[param_tensor].size()):<25}")

except FileNotFoundError:
    print("Không tìm thấy file. Hãy kiểm tra lại đường dẫn save_path.")