# Tên file: models/sampler_ddim.py
# (Cần cài đặt: pip install diffusers)
# === PHIÊN BẢN ĐÃ SỬA: CHẤP NHẬN VÀ TRUYỀN MASK ===

import torch
from diffusers import DDPMScheduler
from tqdm import tqdm

class DDIMSampler:
    """
    Sampler DDIM đã được sửa để chấp nhận và truyền mask 
    vào mô hình Teacher khi sampling.
    """
    def __init__(self, teacher_model, noise_scheduler, null_text_embeddings, null_text_mask):
        self.model = teacher_model
        self.scheduler = noise_scheduler
        self.null_text_embeddings = null_text_embeddings
        self.null_text_mask = null_text_mask # <<< NEW: Cần mask cho null text
        
        self.inference_scheduler = DDPMScheduler(
            num_train_timesteps=noise_scheduler.config.num_train_timesteps,
            beta_schedule=noise_scheduler.config.beta_schedule,
            prediction_type=noise_scheduler.config.prediction_type,
        )
        self.inference_scheduler.config.prediction_type = "epsilon"
        self.model.eval()

    @torch.no_grad()
    def sample(
        self,
        z_1, # Latent noise (t=1) [B, T, D]
        text_embeddings, # [B, L, D]
        text_mask_bool, # <<< NEW: Mask cho text (True = ignore) [B, L]
        pose_mask_bool, # <<< NEW: Mask cho pose (True = ignore) [B, T]
        num_inference_steps=50,
        cfg_scale=7.0,
        device="cuda"
    ):
        """
        Chạy DDIM sampling để tạo ra z0 "sạch" từ z1.
        """
        self.inference_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.inference_scheduler.timesteps
        
        latents = z_1.clone() # z_T
        
        batch_size = latents.shape[0]
        
        # --- SỬA LOGIC CFG ---
        # 1. Chuẩn bị text embeds cho CFG
        uncond_embeds = self.null_text_embeddings.expand(batch_size, -1, -1)
        cond_embeds = text_embeddings
        context_embeds = torch.cat([uncond_embeds, cond_embeds]) # [2*B, L, D]

        # 2. Chuẩn bị text mask cho CFG
        uncond_text_mask = self.null_text_mask.expand(batch_size, -1) # [B, L]
        cond_text_mask = text_mask_bool # [B, L]
        context_text_mask = torch.cat([uncond_text_mask, cond_text_mask]) # [2*B, L]

        # 3. Chuẩn bị pose mask cho CFG
        if pose_mask_bool is not None:
            context_pose_mask = pose_mask_bool.repeat(2, 1) # [2*B, T]
        else:
            context_pose_mask = None
        # --- KẾT THÚC SỬA LOGIC CFG ---

        for t in timesteps:
            latent_model_input = torch.cat([latents] * 2) # [2*B, T, D]
            ts_input = torch.tensor([t] * latent_model_input.shape[0], device=device).long()
            
            # === SỬA DÒNG GỌI MODEL ===
            # Giờ đã truyền mask đầy đủ vào Teacher
            noise_pred = self.model(
                latent_model_input,
                ts_input,
                context_embeds,
                text_mask=context_text_mask, # <<< Đã truyền
                pose_mask=context_pose_mask  # <<< Đã truyền
            )
            # ==========================
            
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred_cfg = noise_pred_uncond + cfg_scale * (noise_pred_text - noise_pred_uncond)
            
            latents = self.inference_scheduler.step(noise_pred_cfg, t, latents).prev_sample
            
        return latents # Đây là pred_z0_teacher