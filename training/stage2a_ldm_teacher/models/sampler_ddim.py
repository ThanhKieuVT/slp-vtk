# Tên file: models/sampler_ddim.py
# (Cần cài đặt: pip install diffusers)

import torch
from diffusers import DDPMScheduler
from tqdm import tqdm

class DDIMSampler:
    """
    Một Sampler DDIM đơn giản để chạy inference cho mô hình Teacher (LDM).
    Nó sẽ được dùng để tạo ra các target "sạch" (z0) cho Student học.
    """
    def __init__(self, teacher_model, noise_scheduler, null_text_embeddings):
        self.model = teacher_model
        self.scheduler = noise_scheduler
        self.null_text_embeddings = null_text_embeddings
        
        # Tạo một scheduler mới chỉ cho inference
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
        num_inference_steps=50,
        cfg_scale=7.0,
        device="cuda"
    ):
        """
        Chạy DDIM sampling (hoặc DDPM) để tạo ra z0 "sạch" từ z1.
        """
        self.inference_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.inference_scheduler.timesteps
        
        latents = z_1.clone() # z_T
        
        # Chuẩn bị text embeddings cho CFG
        batch_size = latents.shape[0]
        uncond_embeds = self.null_text_embeddings.expand(batch_size, -1, -1)
        cond_embeds = text_embeddings
        context_embeds = torch.cat([uncond_embeds, cond_embeds]) # [2*B, L, D]

        # Vòng lặp khử nhiễu
        for t in timesteps:
            # Gấp đôi batch để chạy CFG
            latent_model_input = torch.cat([latents] * 2) # [2*B, T, D]
            
            # Chuẩn bị timestep
            ts_input = torch.tensor([t] * latent_model_input.shape[0], device=device).long()
            
            # Dự đoán noise (model's forward)
            noise_pred = self.model(
                latent_model_input,
                ts_input,
                context_embeds,
                text_mask=None, # Tạm bỏ qua mask khi inference
                pose_mask=None
            )
            
            # Perform CFG
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred_cfg = noise_pred_uncond + cfg_scale * (noise_pred_text - noise_pred_uncond)
            
            # Tính bước DDIM
            # Lưu ý: `scheduler.step` mong đợi `t` là int (chứ không phải tensor)
            latents = self.inference_scheduler.step(noise_pred_cfg, t, latents).prev_sample
            
        return latents # Đây là pred_z0_teacher