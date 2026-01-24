import torch
import torch.nn as nn
from model.YOLO11n_custom import YOLO11_Full

# --- 2. HÀM GỘP THÔNG MINH (Dựa trên logic load_pretrained_weights của bạn) ---
def weld_models(full_model, edge_pt_path, server_pt_path):
    print(f"--- Bắt đầu gộp model ---")
    
    # Load 2 file weights
    # map_location='cpu' để chạy được trên máy không có GPU
    edge_state = torch.load(edge_pt_path, map_location='cpu')
    server_state = torch.load(server_pt_path, map_location='cpu')

    # Xử lý trường hợp file lưu cả optimizer (lấy phần 'model' hoặc 'model_state_dict')
    if 'model_state_dict' in edge_state: edge_state = edge_state['model_state_dict']
    elif 'model' in edge_state: edge_state = edge_state['model'] # Trường hợp lưu dạng Ultralytics

    if 'model_state_dict' in server_state: server_state = server_state['model_state_dict']
    elif 'model' in server_state: server_state = server_state['model']

    full_sd = full_model.state_dict()
    merged_sd = {}
    
    # --- PHASE 1: LOAD EDGE (Layers 0 -> 10) ---
    print("\n>> Đang nạp phần EDGE (Layers 0-10)...")
    for key, value in edge_state.items():
        # Key thường dạng: "layers.0.conv.weight" hoặc "model.0.conv.weight"
        # Ta cần chuẩn hóa về dạng: "layers.0..." để khớp với YOLO11_Full
        
        # 1. Xóa prefix thừa
        clean_key = key.replace('model.', '').replace('layers.', '')
        
        # 2. Tạo key đích
        target_key = f"layers.{clean_key}"
        
        # 3. Kiểm tra xem key này có thuộc 11 layer đầu không
        # Lấy index đầu tiên: "0.conv..." -> lấy số 0
        layer_idx = int(clean_key.split('.')[0])
        
        if layer_idx <= 10: # Chỉ lấy layer thuộc phần Edge
            if target_key in full_sd:
                if full_sd[target_key].shape == value.shape:
                    merged_sd[target_key] = value
                else:
                    print(f"❌ Sai kích thước tại {target_key}: Code {full_sd[target_key].shape} != File {value.shape}")
    
    print(f"   Đã nạp xong phần Edge.")

    # --- PHASE 2: LOAD SERVER (Layers 0 -> 12 của file server --> Maps sang 11 -> 23 của Full) ---
    print("\n>> Đang nạp phần SERVER (Layers 11-23)...")
    SERVER_OFFSET = 11 # Dựa theo code YOLO11_SERVER của bạn
    
    for key, value in server_state.items():
        clean_key = key.replace('model.', '').replace('layers.', '')
        
        # Lấy index cũ trong file server
        parts = clean_key.split('.')
        if parts[0].isdigit():
            old_idx = int(parts[0])
            
            # TÍNH INDEX MỚI CHO FULL MODEL
            new_idx = old_idx + SERVER_OFFSET
            
            # Tạo key mới: thay số cũ bằng số mới
            # Ví dụ: "0.upsample..." -> "11.upsample..."
            new_key_parts = [str(new_idx)] + parts[1:]
            target_key = f"layers.{'.'.join(new_key_parts)}"
            
            if target_key in full_sd:
                if full_sd[target_key].shape == value.shape:
                    merged_sd[target_key] = value
                else:
                    print(f"❌ Sai kích thước tại {target_key} (Gốc {old_idx}->Mới {new_idx}): Code {full_sd[target_key].shape} != File {value.shape}")
            else:
                pass # Bỏ qua các key không khớp (ví dụ key thừa của Detect header cũ)

    print(f"   Đã nạp xong phần Server.")

    # --- PHASE 3: APPLY VÀO MODEL ---
    full_model.load_state_dict(merged_sd, strict=False)
    print("\n✅ HOÀN TẤT! Model Full đã sẵn sàng.")
    return full_model

# --- CHẠY THỰC TẾ ---
# 1. Khởi tạo model full (nhớ sửa config cho đúng channel)
model_full = YOLO11_Full(nc=20)

# 2. Gọi hàm gộp
weld_models(
    model_full,
    edge_pt_path=r"results/run_0_2/client_layer_1_epoch_2.pt",    # File bạn vừa train xong phần đầu
    server_pt_path=r"results/run_0_2/client_layer_2_epoch_2.pt" # File bạn vừa train xong phần sau
)

# 3. Lưu lại model full để dùng sau này đỡ phải gộp lại
torch.save(model_full.state_dict(), "yolo11_merged_full.pt")