# Tên file: inference.py
# === ĐÃ CẬP NHẬT CHO mCLIP (XLM-R) ===

import torch
# === THAY ĐỔI 1: Bỏ CLIP ===
# from transformers import CLIPTextModel, CLIPTokenizer
from transformers import AutoModel, AutoTokenizer
from diffusers import DDPMScheduler
from tqdm import tqdm

# (Import các model của chị)
from models.autoencoder import Stage1Autoencoder
from models.ldm_denoiser import LDM_TransformerDenoiser

@torch.no_grad()
def generate_pose(
    ldm_model, autoencoder, text_encoder, tokenizer,
    prompt,
    guidance_scale=7.5, # Đây là "bí kíp" w > 1
    num_inference_steps=50,
    target_seq_len=100
):
    device = ldm_model.device
    max_len = tokenizer.model_max_length
    
    # 1. Chuẩn bị Text Embeddings (Cond và Uncond)
    text_inputs = tokenizer(
        [prompt], padding="max_length", max_length=max_len, 
        truncation=True, return_tensors="pt"
    ).to(device)
    
    uncond_inputs = tokenizer(
        [""], padding="max_length", max_length=max_len, 
        truncation=True, return_tensors="pt"
    ).to(device)
    
    # === THAY ĐỔI 2: Dùng .last_hidden_state (cho XLM-R) ===
    cond_embeddings = text_encoder(
        text_inputs.input_ids,
        attention_mask=text_inputs.attention_mask
    ).last_hidden_state
    
    uncond_embeddings = text_encoder(
        uncond_inputs.input_ids,
        attention_mask=uncond_inputs.attention_mask
    ).last_hidden_state

    # Lấy attention masks cho LDM model
    cond_text_mask = text_inputs.attention_mask
    uncond_text_mask = uncond_inputs.attention_mask
    
    # 2. Chuẩn bị noise z_T (noise thuần)
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule='squaredcos_cap_v2'
    )
    scheduler.set_timesteps(num_inference_steps)
    
    # Lấy latent_dim từ model một cách an toàn
    latent_dim = ldm_model.latent_dim 
    z_t = torch.randn((1, target_seq_len, latent_dim), device=device)
    
    # 3. Vòng lặp khử nhiễu (Denoising loop)
    for t in tqdm(scheduler.timesteps, desc="Sampling"):
        
        # 4. Chạy model 2 lần (Cond + Uncond)
        latent_model_input = torch.cat([z_t] * 2)
        
        # Mở rộng text embeddings
        text_embed_input = torch.cat([uncond_embeddings, cond_embeddings])
        
        # === THAY ĐỔI 3: Thêm text_mask vào LDM ===
        text_mask_input = torch.cat([uncond_text_mask, cond_text_mask])
        text_padding_mask = ~text_mask_input.bool() # True = ignore
        
        noise_pred = ldm_model(
            latent_model_input,
            t.repeat(2),
            text_embed_input,
            text_mask=text_padding_mask,
            pose_mask=None # Khi inference, pose_mask là None
        )
        
        # Tách kết quả
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        
        # 5. Kỹ thuật: Classifier-Free Guidance (CFG)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        
        # 6. Bước khử nhiễu
        z_t = scheduler.step(noise_pred, t, z_t).prev_sample

    # 7. Cuối cùng, giải mã z0 ra video pose
    final_pose = autoencoder.decode(z_t)
    
    return final_pose # Shape [1, T, 214]