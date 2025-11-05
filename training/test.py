# --- (CELL 2): KIỂM TRA OUTPUT TỪ CHECKPOINT (V4 - Đã fix lỗi Load) ---

import torch
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from accelerate import Accelerator

# 1. Thêm repo vào Python path (Giữ nguyên)
REPO_PATH = '/content/slp-vtk' 
if REPO_PATH not in sys.path:
    sys.path.append(REPO_PATH)
    print(f"✅ Đã thêm {REPO_PATH} vào sys.path")
else:
    print(f"✅ {REPO_PATH} đã có trong sys.path.")

# 2. Import code của chị (Giữ nguyên)
from models.hierarchical_flow import HierarchicalFlowMatcher
from data.phoenix_dataset import PhoenixFlowDataset, collate_fn

print("Đang khởi tạo model và data...")

try:
    # 3. Khởi tạo Accelerator (Giữ nguyên)
    accelerator = Accelerator(mixed_precision='fp16')

    # 4. Định nghĩa đường dẫn checkpoint (Giữ nguyên)
    CHECKPOINT_EPOCH = "50" # <--- Chị có thể sửa số epoch ở đây
    
    CHECKPOINT_NAME = f"checkpoint_epoch_{CHECKPOINT_EPOCH}.pt"
    CHECKPOINT_PATH = os.path.join(os.environ['OUTPUT_ROOT'], 'manual_flow', CHECKPOINT_NAME)
    DATA_ROOT = os.environ['DATA_ROOT']

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"LỖI: Không tìm thấy checkpoint tại: {CHECKPOINT_PATH}")
    else:
        print(f"Sẽ load checkpoint từ: {CHECKPOINT_PATH}")

        # 5. Load Model (ĐÃ SỬA LỖI)
        # --- (Bắt đầu phần code mới) ---
        
        # Khởi tạo model TRƯỚC
        # (Lưu ý: Nếu Class này cần tham số config, chị phải điền vào đây)
        model = HierarchicalFlowMatcher() 

        # Load state dict TỪ FILE .pt
        print(f"Đang load state_dict từ file: {CHECKPOINT_PATH}")
        # Dùng map_location='cpu' để an toàn, tránh load thẳng lên GPU
        state_dict = torch.load(CHECKPOINT_PATH, map_location='cpu')

        # KIỂM TRA: Đôi khi file .pt là 1 dict { 'model': ..., 'epoch': ... }
        # Ta cần lấy đúng state_dict của model
        if 'model' in state_dict:
            state_dict = state_dict['model']
            print("Đã tìm thấy key 'model' trong checkpoint.")
        elif 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
            print("Đã tìm thấy key 'model_state_dict' trong checkpoint.")

        # Giờ mới prepare model
        model = accelerator.prepare(model)

        # Load state_dict vào model
        # (Phải unwrap model trước khi load, vì state_dict này là của model gốc)
        try:
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.load_state_dict(state_dict)
            print("✅ Model đã load weights thành công.")
        except Exception as e:
            print(f"LỖI khi load state_dict: {e}")
            print("Có thể state_dict không khớp. Đang thử load trực tiếp...")
            model.load_state_dict(state_dict) # Thử cách dự phòng
            print("✅ Model đã load weights (theo cách dự phòng).")

        model.eval() # QUAN TRỌNG: Chuyển sang chế độ evaluation
        # --- (Kết thúc phần code mới) ---

        # 6. Load Dataset (Giữ nguyên)
        # (Lưu ý: Nếu Class này cần tham số khác, chị phải sửa ở đây)
        val_dataset = PhoenixFlowDataset(data_root=DATA_ROOT, split='val') 

        # Lấy 1 sample (ví dụ: sample số 0)
        batch = val_dataset[0] # Lấy sample đầu tiên trong tập validation
        print(f"✅ Đã load sample 0 từ dataset.")

        # 7. Chuẩn bị Batch 
        # --- [CHỊ CẦN SỬA - QUAN TRỌNG NHẤT] ---
        # Mở file 'data/phoenix_dataset.py', xem hàm `__getitem__`
        # nó trả về `dict` với các key là gì?
        # Thay 'input_key' và 'gt_key' bằng các key thật đó.
        
        # Ví dụ: Nếu key thật là 'video_frames' và 'flow_target'
        # input_data = batch['video_frames'].unsqueeze(0)
        # ground_truth = batch['flow_target'].unsqueeze(0)
        
        input_data = batch['input_key'].unsqueeze(0)    # <--- THAY 'input_key' NÀY
        ground_truth = batch['gt_key'].unsqueeze(0) # <--- THAY 'gt_key' NÀY
        # .unsqueeze(0) là để thêm chiều batch_size=1
        # ------------------------------------------------
        
        # Đưa data lên device (GPU)
        input_data, ground_truth = accelerator.prepare(input_data, ground_truth)
        
        print(f"Input shape (đã thêm batch): {input_data.shape}")
        print(f"Ground truth shape (đã thêm batch): {ground_truth.shape}")

        # 8. Chạy Inference
        print("Đang chạy inference...")
        with torch.no_grad(): 
            # --- [CÓ THỂ CẦN SỬA] ---
            # Nếu model(input) trả về một dict, chị phải lấy đúng key dự đoán
            # Ví dụ: prediction = model(input_data)['pred_flow']
            prediction = model(input_data) # <--- Đây là output của model

        # 9. So sánh kết quả
        prediction_np = prediction.cpu().float().numpy()
        ground_truth_np = ground_truth.cpu().float().numpy()

        print("\n--- KẾT QUẢ DỰ ĐOÁN (PREDICTION) ---")
        print(f"Shape: {prediction_np.shape} | Min: {prediction_np.min():.4f} | Max: {prediction_np.max():.4f} | Mean: {prediction_np.mean():.4f}")

        print("\n--- KẾT QUẢ GROUND TRUTH ---")
        print(f"Shape: {ground_truth_np.shape} | Min: {ground_truth_np.min():.4f} | Max: {ground_truth_np.max():.4f} | Mean: {ground_truth_np.mean():.4f}")

        # 10. (Optional) Visualize nếu là ảnh/flow
        if len(prediction_np.shape) == 4: # Giả sử [B, C, H, W]
            pred_img = prediction_np[0, 0] # Lấy channel đầu tiên
            gt_img = ground_truth_np[0, 0] # Lấy channel đầu tiên
            
            print("\nĐang visualize channel 0 của sample 0...")
            fig, ax = plt.subplots(1, 2, figsize=(16, 8))
            
            im0 = ax[0].imshow(pred_img, cmap='gray')
            ax[0].set_title(f"Dự đoán (Prediction)\nShape: {pred_img.shape}")
            ax[0].axis('off')
            fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
            
            im1 = ax[1].imshow(gt_img, cmap='gray')
            ax[1].set_title(f"Ground Truth\nShape: {gt_img.shape}")
            ax[1].axis('off')
            fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
            
            plt.suptitle(f"So sánh Output từ Epoch {CHECKPOINT_EPOCH}", fontsize=16)
            plt.show()

except KeyError as e:
    print("\n--- LỖI ---")
    print(f"Gặp lỗi KeyError: {e}")
    print("CHÚ Ý: Chị đã nhập sai 'key' cho batch data ở bước 7.")
    print("Vui lòng mở file 'data/phoenix_dataset.py' để xem key đúng.")
except Exception as e:
    print(f"\n--- LỖI KHÁC ---")
    print(f"Gặp lỗi không xác định: {e}")
    print("CHÚ Ý: Lỗi có thể ở bước 5, 6 (thiếu tham số) hoặc bước 8 (output model là dict).")