# Tên file: inference.py
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import DDPMScheduler

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
    
    # 1. Chuẩn bị Text Embeddings (Cond và Uncond)
    text_inputs = tokenizer(
        [prompt], padding="max_length", max_length=tokenizer.model_max_length, 
        truncation=True, return_tensors="pt"
    ).to(device)
    
    uncond_inputs = tokenizer(
        [""], padding="max_length", max_length=tokenizer.model_max_length, 
        truncation=True, return_tensors="pt"
    ).to(device)
    
    cond_embeddings = text_encoder(
        text_inputs.input_ids,
        attention_mask=text_inputs.attention_mask
    )[0]
    
    uncond_embeddings = text_encoder(
        uncond_inputs.input_ids,
        attention_mask=uncond_inputs.attention_mask
    )[0]
    
    # 2. Chuẩn bị noise z_T (noise thuần)
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule='squaredcos_cap_v2'
    )
    scheduler.set_timesteps(num_inference_steps)
    
    z_t = torch.randn((1, target_seq_len, ldm_model.latent_dim), device=device)
    
    # 3. Vòng lặp khử nhiễu (Denoising loop)
    for t in tqdm(scheduler.timesteps, desc="Sampling"):
        
        # 4. Chạy model 2 lần (Cond + Uncond)
        
        # Mở rộng z_t để chạy 2 mẫu song song
        latent_model_input = torch.cat([z_t] * 2)
        
        # Mở rộng text
        text_embed_input = torch.cat([uncond_embeddings, cond_embeddings])
        
        # Mask (nếu có, ở đây ví dụ không cần)
        
        noise_pred = ldm_model(
            latent_model_input,
            t.repeat(2),
            text_embed_input
        )
        
        # Tách kết quả
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        
        # 5. Kỹ thuật: Classifier-Free Guidance (CFG)
        # noise_pred = uncond + w * (cond - uncond)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        
        # 6. Bước khử nhiễu
        z_t = scheduler.step(noise_pred, t, z_t).prev_sample

    # 7. Cuối cùng, giải mã z0 ra video pose
    final_pose = autoencoder.decode(z_t)
    
    return final_pose # Shape [1, T, 214]