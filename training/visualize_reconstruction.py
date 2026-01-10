"""
OpenCV Visualization for Sign Language Pose
Style: Ellipsoid Bones on White Background (Similar to Sign-IDD)
"""
import numpy as np
import cv2
import math
import argparse
import sys

# --- COLORS (BGR for OpenCV) ---
COLOR_BODY = (0, 0, 0)       # Black
COLOR_HANDS = (81, 101, 21)  # Teal (BGR of #156551)
COLOR_MOUTH = (43, 57, 192)  # Red (BGR of #c0392b)

# --- TOPOLOGY ---
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20)
]
POSE_CONNECTIONS_UPPER_BODY = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), # Shoulders -> Arms -> Wrists
    (11, 23), (12, 24), (23, 24), # Torso Box
    (1, 0), (0, 2), (0, 5) # Neck/Nose/Eyes approx
]
FACE_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8)]

MOUTH_OUTER_LIP = list(zip(range(0, 11), range(1, 12))) + [(11, 0)]
MOUTH_INNER_LIP = list(zip(range(12, 19), range(13, 20))) + [(19, 12)]
MOUTH_CONNECTIONS_20 = MOUTH_OUTER_LIP + MOUTH_INNER_LIP

ALL_CONN = []
# Body
ALL_CONN.extend([{'indices': (s, e), 'offset': 0, 'color': COLOR_BODY, 'width': 3} for (s, e) in POSE_CONNECTIONS_UPPER_BODY])
ALL_CONN.extend([{'indices': (s, e), 'offset': 0, 'color': COLOR_BODY, 'width': 2} for (s, e) in FACE_CONNECTIONS])

# Hands (Teal)
ALL_CONN.extend([{'indices': (s, e), 'offset': 33, 'color': COLOR_HANDS, 'width': 2} for (s, e) in HAND_CONNECTIONS])
ALL_CONN.append({'indices': (15, 0), 'offset': (0, 33), 'color': COLOR_HANDS, 'width': 2})
ALL_CONN.extend([{'indices': (s, e), 'offset': 54, 'color': COLOR_HANDS, 'width': 2} for (s, e) in HAND_CONNECTIONS])
ALL_CONN.append({'indices': (16, 0), 'offset': (0, 54), 'color': COLOR_HANDS, 'width': 2})

# Mouth (Red)
ALL_CONN.extend([{'indices': (s, e), 'offset': 75, 'color': COLOR_MOUTH, 'width': 1} for (s, e) in MOUTH_CONNECTIONS_20])


class DataProcessor:
    def process_sequence(self, raw_data):
        # Extract 95 points
        kps = self._extract_95_points(raw_data)
        
        # Clean NaN (<0.1)
        kps[np.abs(kps) < 0.1] = np.nan
        
        # Fill NaN (Simple/Linear) - basic logic without pandas to reduce deps if desired
        # But using pandas is safer for smooth interpolation
        try:
            import pandas as pd
            for i in range(kps.shape[1]):
                for c in range(2):
                    s = pd.Series(kps[:, i, c])
                    kps[:, i, c] = s.interpolate(method='linear', limit_direction='both').to_numpy()
        except:
            pass # Skip interpolation if pandas missing
            
        # Wrist Gluing
        kps[:, 33, :] = kps[:, 15, :]
        kps[:, 54, :] = kps[:, 16, :]
        
        return kps

    def _extract_95_points(self, data):
        if isinstance(data, dict):
            data = list(data.values())[0] if 'pose' not in data and 'keypoints' not in data else (data['keypoints'] if 'keypoints' in data else data['pose'])
        elif hasattr(data, 'files'): data = data['keypoints']
        
        # Convert to numpy
        data = np.array(data)
        
        if data.ndim == 2:
            T, D = data.shape
            if D == 214:
                manual = data[:, :150].reshape(T, 75, 2)
                mouth = data[:, 174:].reshape(T, 20, 2)
                return np.concatenate([manual, mouth], axis=1)
            elif D == 150:
                return data.reshape(T, 75, 2)
        elif data.ndim == 3 and data.shape[1] == 75:
             return data
             
        return data.reshape(len(data), -1, 2)


def draw_line(im, joint1, joint2, c=(0, 0, 0), width=3):
    if np.isnan(joint1).any() or np.isnan(joint2).any(): return
    
    center = (int((joint1[0] + joint2[0]) / 2), int((joint1[1] + joint2[1]) / 2))
    length = int(math.sqrt(((joint1[0] - joint2[0]) ** 2) + ((joint1[1] - joint2[1]) ** 2)) / 2)
    angle = math.degrees(math.atan2((joint1[0] - joint2[0]), (joint1[1] - joint2[1])))
    
    cv2.ellipse(im, center, (width, length), -angle, 0.0, 360.0, c, -1)


def render_video(gt_path, recon_path, output_path):
    print("Loading...")
    gt_214 = np.load(gt_path, allow_pickle=True)
    recon_214 = np.load(recon_path, allow_pickle=True)
    
    proc = DataProcessor()
    gt_kps = proc.process_sequence(gt_214)
    recon_kps = proc.process_sequence(recon_214)
    
    T = min(len(gt_kps), len(recon_kps))
    gt_kps = gt_kps[:T]
    recon_kps = recon_kps[:T]
    
    print(f"Rendering {T} frames via OpenCV...")
    
    # Auto-scale logic
    # Find bounds of GT
    all_pts = gt_kps.reshape(-1, 2)
    valid_pts = all_pts[~np.isnan(all_pts).any(axis=1)]
    if len(valid_pts) > 0:
        min_xy = np.percentile(valid_pts, 1, axis=0)
        max_xy = np.percentile(valid_pts, 99, axis=0)
        center = (min_xy + max_xy) / 2
        
        # Scaling factor to fit 650x650 with padding
        span = max_xy - min_xy
        max_span = max(span[0], span[1])
        scale = 500 / max_span # Leave 75px padding
        
        # Shift to center (325, 325)
        offset = np.array([325, 325]) - center * scale
    else:
        scale = 300
        offset = np.array([325, 325])
        
    def to_pixel(pt):
        # Apply scaling and offset
        # Note: Y needs split? No, input is already standard. 
        # Matplotlib uses inverted Y. CV2 uses top-left origin.
        # If input data has Y UP, we need to invert. Usually pose data is Y DOWN (image coords) or Y UP (world).
        # Assuming Y UP from normalization? 
        # Actually usually normalized data is centered 0.
        # Let's assume standard orientation (we might need to flip Y in scale)
        
        p = pt * scale
        # Flip Y if needed (User said "inverted axis" in previous matplot code)
        # Matplotlib `invert_yaxis()` was used. So data is likely Image Coords (Y down).
        # Valid range [-1, 1] usually.
        # If invert_yaxis was used, then Y increases downwards in plot logic?
        # Typically pose data from videos is Y-down.
        # Let's try direct mapping. If upside down, we flip.
        
        return (int(p[0] + offset[0]), int(p[1] + offset[1]))

    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, 25.0, (1300, 650)) # Double width for SBS
    
    has_mouth = gt_kps.shape[1] >= 95
    
    for i in range(T):
        frame = np.ones((650, 1300, 3), dtype=np.uint8) * 255
        
        # GT (Left)
        # Draw Separator
        cv2.line(frame, (650, 0), (650, 650), (0,0,0), 2)
        cv2.putText(frame, "GROUND TRUTH", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
        cv2.putText(frame, "RECONSTRUCTED", (850, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
        
        for kps_seq, x_shift in [(gt_kps, 0), (recon_kps, 650)]:
            pose = kps_seq[i]
            
            # Draw Bones
            for item in ALL_CONN:
                if item['offset'] == 75 and not has_mouth: continue
                
                s, e = item['indices']
                off = item['offset']
                if isinstance(off, tuple): s, e = s+off[0], e+off[1]
                else: s, e = s+off, e+off
                
                if s < len(pose) and e < len(pose):
                    p1 = pose[s]
                    p2 = pose[e]
                    
                    if not (np.isnan(p1).any() or np.isnan(p2).any()):
                        # Apply scale
                        px1 = (int(p1[0] * scale + offset[0] + x_shift), int(p1[1] * scale + offset[1]))
                        px2 = (int(p2[0] * scale + offset[0] + x_shift), int(p2[1] * scale + offset[1]))
                        
                        draw_line(frame, px1, px2, item['color'], item['width'])

        video.write(frame)
        
    video.release()
    print(f"✅ Saved: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', required=True)
    parser.add_argument('--recon', required=True)
    parser.add_argument('--output', default='comparison_cv2.mp4')
    args = parser.parse_args()
    
    render_video(args.gt, args.recon, args.output)