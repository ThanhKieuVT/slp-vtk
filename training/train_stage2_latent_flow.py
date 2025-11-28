def estimate_scale_factor(encoder, dataloader, device, max_samples=1024):
    encoder.eval()
    latents = []
    seen = 0
    print("Computing latent scale factor...")
    with torch.no_grad():
        for batch in dataloader:
            poses = None
            if isinstance(batch, dict):
                if "poses" in batch: poses = batch["poses"]
                elif "pose" in batch: poses = batch["pose"]
                elif "x" in batch: poses = batch["x"]
            else:
                poses = batch[0]
            
            if poses is None: continue
                
            poses = poses.to(device)
            z = encoder.encode(poses)
            latents.append(z.detach().cpu())
            seen += z.shape[0]
            if seen >= max_samples:
                break
    
    if not latents: return 1.0
    latents = torch.cat(latents, dim=0)
    
    # FIX #5: Dùng Quantile để tránh outlier làm hỏng scale
    # Lấy giá trị tại percentile 95% của trị tuyệt đối
    scale = float(latents.abs().quantile(0.95))
    
    # Fallback nếu scale quá nhỏ
    return max(scale, 1e-6)