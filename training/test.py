# --- (CELL 2): KIỂM TRA OUTPUT TỪ CHECKPOINT (V2 - Đã fix import) ---

import torch
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from accelerate import Accelerator

# 1. [FIX] Thêm repo vào Python path (để import được code của chị)
# Lỗi "ModuleNotFoundError" là do thiếu bước này.
REPO_PATH = '/content/slp-vtk' 
if REPO_PATH not in sys.path:
    sys.path.append(REPO_PATH)
    print(f"✅ Đã thêm {REPO_PATH} vào sys.path")
else:
    print(f"✅ {REPO_PATH} đã có trong sys.path.")

# -----------------------------------------------------------------
# --- [CHỊ CẦN SỬA] ---
# 1. Mở file 'training/train_manual_flow.py'
# 2. XÓA 2 DÒNG VÍ DỤ CỦA EM BÊN DƯỚI
# 3. DÁN 2 DÒNG IMPORT THẬT CỦA CHỊ VÀO
#
# --- XÓA 2 DÒNG NÀY ĐI ---
from models.hierarchical_flow import HierarchicalFlowMatcher
from data.phoenix_dataset import PhoenixFlowDataset, collate_fn
#
# --- DÁN 2 DÒNG THẬT CỦA CHỊ VÀO ĐÂY ---
# (Ví dụ: from models.my_model import MyAwesomeModel)
# (Ví dụ: from dataset.my_data import MyDataset)
# -----------------------------------------------------------------


print("Đang khởi tạo model và data...")

try:
    # 2. Khởi tạo Accelerator (giống trong code train)
    accelerator = Accelerator(mixed_precision='fp16')

    # 3. Định nghĩa đường dẫn checkpoint
    # --- [CHỊ CẦN SỬA] ---
    # Thay "50" bằng số epoch chị muốn kiểm tra
    CHECKPOINT_EPOCH = "50" # <--- SỬA SỐ EPOCH Ở ĐÂY
    # -----------------------
    
    CHECKPOINT_NAME = f"checkpoint_epoch_{CHECKPOINT_EPOCH}.pt"
    CHECKPOINT_PATH = os.path.join(os.environ['OUTPUT_ROOT'], 'manual_flow', CHECKPOINT_NAME)
    DATA_ROOT = os.environ['DATA_ROOT']

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"LỖI: Không tìm thấy checkpoint tại: {CHECKPOINT_PATH}")
        print("Vui lòng kiểm tra lại CHECKPOINT_EPOCH.")
    else:
        print(f"Sẽ load checkpoint từ: {CHECKPOINT_PATH}")

        # 4. Load Model
        # --- [CHỊ CẦN SỬA] ---
        # Thay thế "ManualFlowModel()" bằng cách khởi tạo model của chị
        model = HierarchicalFlowMatcher() # <--- THAY THẾ BẰNG CLASS MODEL CỦA CHỊ
        # -----------------------
        
        model = accelerator.prepare(model)
        
        # Load state dict
        accelerator.load_state(CHECKPOINT_PATH)
        print("✅ Model đã load weights từ checkpoint.")
        model.eval() # QUAN TRỌNG: Chuyển sang chế độ evaluation

        # 5. Load Dataset (chỉ cần 1 sample từ val/test)
        # --- [CHỊ CẦN SỬA] ---
        # Thay thế "PhoenixDataset()" bằng cách khởi tạo dataset của chị
        val_dataset = PhoenixDataset(data_root=DATA_ROOT, split='val') # <--- THAY THẾ
        # -----------------------

        # Lấy 1 sample (ví dụ: sample số 0)
        batch = val_dataset[0] 
        print(f"✅ Đã load sample 0 từ dataset.")

        # 6. Chuẩn bị Batch
        # --- [CHỊ CẦN SỬA] ---
        # Em giả sử batch là một DICT.
        # Chị thay 'input_key' và 'gt_key' bằng các key chính xác.
        
        input_data = batch['input_key'].unsqueeze(0)    # <--- THAY 'input_key'
        ground_truth = batch['gt_key'].unsqueeze(0) # <--- THAY 'gt_key'
        # .unsqueeze(0) là để thêm chiều batch_size=1
        # -----------------------
        
        # Đưa data lên device (GPU)
        input_data, ground_truth = accelerator.prepare(input_data, ground_truth)
        
        print(f"Input shape (đã thêm batch): {input_data.shape}")
        print(f"Ground truth shape (đã thêm batch): {ground_truth.shape}")

        # 7. Chạy Inference
        print("Đang chạy inference...")
        with torch.no_grad(): # Không cần tính gradient
            prediction = model(input_data) # <--- Đây là output của model

        # 8. So sánh kết quả
        prediction_np = prediction.cpu().float().numpy()
        ground_truth_np = ground_truth.cpu().float().numpy()

        print("\n--- KẾT QUẢ DỰ ĐOÁN (PREDICTION) ---")
        print(f"Shape: {prediction_np.shape} | Min: {prediction_np.min():.4f} | Max: {prediction_np.max():.4f} | Mean: {prediction_np.mean():.4f}")

        print("\n--- KẾT QUẢ GROUND TRUTH ---")
        print(f"Shape: {ground_truth_np.shape} | Min: {ground_truth_np.min():.4f} | Max: {ground_truth_np.max():.4f} | Mean: {ground_truth_np.mean():.4f}")

        # 9. (Optional) Visualize nếu là ảnh/flow
        if len(prediction_np.shape) == 4:
            pred_img = prediction_np[0, 0] 
            gt_img = ground_truth_np[0, 0]
            
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

except NameError as e:
    print("\n--- LỖI ---")
    print(f"Gặp lỗi: {e}")
    print("CHÚ Ý: Rất có thể chị chưa sửa tên Class Model hoặc Dataset.")
    print("Vui lòng kiểm tra lại phần [CHỊ CẦN SỬA] ở đầu cell.")
except KeyError as e:
    print("\n--- LỖI ---")
    print(f"Gặp lỗi KeyError: {e}")
    print("CHÚ Ý: Rất có thể chị đã nhập sai 'key' cho batch data ở bước 6.")
except Exception as e:
    print(f"\n--- LỖI KHÁC ---")
    print(f"Gặp lỗi không xác định: {e}")