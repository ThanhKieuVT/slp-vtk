"""
Script: Visualize Original vs Reconstructed Pose
Tạo video side-by-side để so sánh
"""

import numpy as np
import cv2
import argparse
import os
from tqdm import tqdm

# Phoenix-2014T keypoint connections (75 keypoints = 25*3: body, left_hand, right_hand)
BODY_CONNECTIONS = [
    # Torso
    (0, 1), (1, 2), (2, 3), (3, 4),  # Spine
    # Left arm
    (1, 5), (5, 6), (6, 7),
    # Right arm
    (1, 8), (8, 9), (9, 10),
    # Left leg
    (0, 11), (11, 12), (12, 13),
    # Right leg
    (0, 14), (14, 15), (15, 16),
    # Head
    (0, 17), (17, 18), (18, 19), (19, 20),
]

HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
]


def normalize_coords(coords, width, height):
    """Normalize coordinates to image size"""
    coords_norm = coords.copy()
    coords_norm[:, 0] = coords[:, 0] * width
    coords_norm[:, 1] = coords[:, 1] * height
    return coords_norm.astype(np.int32)


def draw_skeleton(frame, keypoints, color=(0, 255, 0), radius=3):
    """
    Draw skeleton on frame
    keypoints: [75, 2] array (x, y coordinates)
    """
    # Split into body, left_hand, right_hand
    body = keypoints[:25]
    left_hand = keypoints[25:46]
    right_hand = keypoints[46:67]
    
    # Draw body
    for connection in BODY_CONNECTIONS:
        if connection[0] < len(body) and connection[1] < len(body):
            pt1_arr = body[connection[0]]
            pt2_arr = body[connection[1]]
            # Check if both points are valid (not zero or negative)
            if pt1_arr[0] > 0 and pt1_arr[1] > 0 and pt2_arr[0] > 0 and pt2_arr[1] > 0:
                pt1 = (int(pt1_arr[0]), int(pt1_arr[1]))
                pt2 = (int(pt2_arr[0]), int(pt2_arr[1]))
                cv2.line(frame, pt1, pt2, color, 2)
    
    # Draw hands
    for hand, hand_keypoints in [("left", left_hand), ("right", right_hand)]:
        for connection in HAND_CONNECTIONS:
            if connection[0] < len(hand_keypoints) and connection[1] < len(hand_keypoints):
                pt1_arr = hand_keypoints[connection[0]]
                pt2_arr = hand_keypoints[connection[1]]
                if pt1_arr[0] > 0 and pt1_arr[1] > 0 and pt2_arr[0] > 0 and pt2_arr[1] > 0:
                    pt1 = (int(pt1_arr[0]), int(pt1_arr[1]))
                    pt2 = (int(pt2_arr[0]), int(pt2_arr[1]))
                    cv2.line(frame, pt1, pt2, color, 1)
    
    # Draw keypoints
    for kp in keypoints:
        if kp[0] > 0 and kp[1] > 0:
            cv2.circle(frame, (int(kp[0]), int(kp[1])), radius, color, -1)
    
    return frame


def create_comparison_video(original_path, reconstructed_path, output_video, fps=25):
    """
    Create side-by-side comparison video
    """
    # Load data
    original = np.load(original_path)  # [T, 214]
    reconstructed = np.load(reconstructed_path)  # [T, 214]
    
    T = len(original)
    
    # Extract manual features (150 dims = 75 keypoints * 2)
    original_kp = original[:, :150].reshape(T, 75, 2)
    recon_kp = reconstructed[:, :150].reshape(T, 75, 2)
    
    # Video settings
    width, height = 640, 480
    frame_width = width * 2  # Side by side
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, height))
    
    print(f"Creating video with {T} frames...")
    
    for t in tqdm(range(T)):
        # Create blank frames
        frame_orig = np.zeros((height, width, 3), dtype=np.uint8)
        frame_recon = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Normalize coordinates
        kp_orig = normalize_coords(original_kp[t], width, height)
        kp_recon = normalize_coords(recon_kp[t], width, height)
        
        # Draw skeletons
        frame_orig = draw_skeleton(frame_orig, kp_orig, color=(0, 255, 0))
        frame_recon = draw_skeleton(frame_recon, kp_recon, color=(0, 0, 255))
        
        # Add labels
        cv2.putText(frame_orig, "Original", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame_recon, "Reconstructed", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Frame number
        cv2.putText(frame_orig, f"Frame: {t+1}/{T}", (10, height-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Concatenate side by side
        frame_combined = np.hstack([frame_orig, frame_recon])
        
        # Write frame
        out.write(frame_combined)
    
    out.release()
    print(f"✅ Video saved: {output_video}")


def compute_per_frame_error(original_path, reconstructed_path):
    """Compute MSE/MAE per frame"""
    original = np.load(original_path)
    reconstructed = np.load(reconstructed_path)
    
    T = len(original)
    
    # Compute errors per frame
    mse_per_frame = np.mean((original - reconstructed) ** 2, axis=1)
    mae_per_frame = np.mean(np.abs(original - reconstructed), axis=1)
    
    print("\n📊 Per-Frame Error Statistics:")
    print(f"   MSE: mean={mse_per_frame.mean():.6f}, std={mse_per_frame.std():.6f}")
    print(f"   MSE: min={mse_per_frame.min():.6f}, max={mse_per_frame.max():.6f}")
    print(f"   MAE: mean={mae_per_frame.mean():.6f}, std={mae_per_frame.std():.6f}")
    print(f"   MAE: min={mae_per_frame.min():.6f}, max={mae_per_frame.max():.6f}")
    
    # Find worst frames
    worst_frames = np.argsort(mse_per_frame)[-5:]
    print(f"\n⚠️  5 Worst frames (highest MSE):")
    for idx in worst_frames:
        print(f"   Frame {idx}: MSE={mse_per_frame[idx]:.6f}, MAE={mae_per_frame[idx]:.6f}")
    
    return mse_per_frame, mae_per_frame


def main():
    parser = argparse.ArgumentParser(description='Visualize Reconstruction Quality')
    parser.add_argument('--original', type=str, required=True,
                       help='Path to original .npy file')
    parser.add_argument('--reconstructed', type=str, required=True,
                       help='Path to reconstructed .npy file')
    parser.add_argument('--output_video', type=str, default='comparison.mp4',
                       help='Output video path')
    parser.add_argument('--fps', type=int, default=25,
                       help='Video FPS')
    parser.add_argument('--show_errors', action='store_true',
                       help='Show per-frame errors')
    
    args = parser.parse_args()
    
    # Check files exist
    if not os.path.exists(args.original):
        print(f"❌ Original file not found: {args.original}")
        return
    
    if not os.path.exists(args.reconstructed):
        print(f"❌ Reconstructed file not found: {args.reconstructed}")
        return
    
    print("🎬 Creating comparison video...")
    create_comparison_video(
        args.original,
        args.reconstructed,
        args.output_video,
        fps=args.fps
    )
    
    if args.show_errors:
        compute_per_frame_error(args.original, args.reconstructed)
    
    print("\n✅ Done!")
    print(f"💡 Play video: vlc {args.output_video}")


if __name__ == '__main__':
    main()