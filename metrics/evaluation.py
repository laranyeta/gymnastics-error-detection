import json
import numpy as np
import pandas as pd

coco2mp = { #coco to mediapipe format (17 keypoints to 33 keypoints)
    5: 11, 6: 12, 7: 13, 8: 14, 9: 15, 10: 16,
    11: 23, 12: 24, 13: 25, 14: 26, 15: 27, 16: 28
}

KAPPAS = { #weights without face bones as they are not relevant (OKS standard values)
    11: 0.079, 12: 0.079, 13: 0.072, 14: 0.072, 15: 0.062, 16: 0.062,
    23: 0.107, 24: 0.107, 25: 0.087, 26: 0.087, 27: 0.089, 28: 0.089
}

def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    if len(data) > 0 and isinstance(data[0], dict) and "position" in data[0]: #test data has "position" but original doesn't
        return pd.DataFrame([frame["position"] for frame in data])
    return pd.DataFrame(data)

def normalize_dataframe(df): #normalize from torso
    torso = ['x_11', 'x_12', 'x_23', 'x_24']
    if not all(col in df.columns for col in torso):
        return df

    neck_x = (df['x_11'] + df['x_12']) / 2
    neck_y = (df['y_11'] + df['y_12']) / 2
    pelvis_x = (df['x_23'] + df['x_24']) / 2
    pelvis_y = (df['y_23'] + df['y_24']) / 2
    
    mid_torso_x = (neck_x + pelvis_x) / 2
    mid_torso_y = (neck_y + pelvis_y) / 2
    
    d = np.sqrt((neck_x - pelvis_x)**2 + (neck_y - pelvis_y)**2).replace(0, 1) #distance from torso
    df_norm = df.copy()
    for i in range(33):
        if f'x_{i}' in df.columns:
            df_norm[f'x_{i}'] = (df[f'x_{i}'] - mid_torso_x) / d
            df_norm[f'y_{i}'] = (df[f'y_{i}'] - mid_torso_y) / d
    return df_norm

def get_raw_pixels_from_json(json_path, is_coco=True):
    with open(json_path, 'r') as f:
        data = json.load(f)
    if is_coco: #running datasets
        sorted_imgs = sorted(data['images'], key=lambda x: x['file_name'])
    else: #test dataset
        sorted_imgs = sorted(data['images'], key=lambda x: x['id'])
    images_map = {img['id']: i for i, img in enumerate(sorted_imgs)}

    num_frames = len(images_map)
    raw_frames = [{f"{ax}_{idx}": np.nan for idx in range(33) for ax in ['x', 'y']} for _ in range(num_frames)]
    
    for ann in data.get('annotations', []):
        frame_idx = images_map.get(ann['image_id'])
        if frame_idx is None: continue
        
        kpts = ann['keypoints'] 
        for coco_idx, mp_idx in coco2mp.items(): #[x,y,visibility]
            if len(kpts) > coco_idx * 3 + 2 and kpts[coco_idx * 3 + 2] > 0:
                raw_frames[frame_idx][f"x_{mp_idx}"] = kpts[coco_idx * 3]
                raw_frames[frame_idx][f"y_{mp_idx}"] = kpts[coco_idx * 3 + 1]
    return pd.DataFrame(raw_frames)

### EVALUATION METRICS
def map(gt, pred, thr=0.5):
    results = {}
    for mp_idx in coco2mp.values():
        kappa = KAPPAS[mp_idx]
        dx = gt[f'x_{mp_idx}'] - pred[f'x_{mp_idx}']
        dy = gt[f'y_{mp_idx}'] - pred[f'y_{mp_idx}']
        distances_sq = (dx**2 + dy**2)

        oks_scores = np.exp(-distances_sq / (2 * 1.0 * (kappa**2))) 
        valid_frames = gt[f'x_{mp_idx}'].notna()
        if valid_frames.sum() > 0:
            results[f'Point_{mp_idx}'] = (oks_scores[valid_frames] >= thr).mean()
        else:
            results[f'Point_{mp_idx}'] = np.nan
            
    results['MEAN_mAP'] = np.nanmean(list(results.values()))
    return results

def pck(gt, pred, thr=0.2):
    results = {}
    for mp_idx in coco2mp.values():
        dx = gt[f'x_{mp_idx}'] - pred[f'x_{mp_idx}']
        dy = gt[f'y_{mp_idx}'] - pred[f'y_{mp_idx}']
        distances = np.sqrt(dx**2 + dy**2) 
        
        valid_frames = gt[f'x_{mp_idx}'].notna()
        if valid_frames.sum() > 0:
            results[f'Point_{mp_idx}'] = (distances[valid_frames] <= thr).mean()
        else:
            results[f'Point_{mp_idx}'] = np.nan
            
    results['MEAN_PCK'] = np.nanmean(list(results.values()))
    return results

if __name__ == "__main__":
    print("Which dataset do you want to use?\n")
    print("1: Customized Roboflow Dataset (Running)")
    print("2: Gymnastics Beam Balance (Test)")
    dataset_opt = input("> ")
        
    if dataset_opt == "1": #running dataset
            GT_PATH = "dataset/evaluate/running/groundtruth.json"
            MP_PATH = "dataset/evaluate/running/mediapipe.json"
            SP_PATH = "dataset/evaluate/running/sapiens.json"
            YO_PATH = "dataset/evaluate/running/yolo.json"
            
            gt = normalize_dataframe(get_raw_pixels_from_json(GT_PATH, is_coco=True))
            mediapipe = normalize_dataframe(load_json(MP_PATH))
            sapiens = normalize_dataframe(load_json(SP_PATH))
            yolo = normalize_dataframe(load_json(YO_PATH))
    
    elif dataset_opt == "2": #test
            GT_PATH = "dataset/evaluate/test/groundtruth.json"
            MP_PATH = "dataset/evaluate/test/mediapipe.json"
            SP_PATH = "dataset/evaluate/test/sapiens.json"
            YO_PATH = "dataset/evaluate/test/yolo.json"
            
            gt_raw = get_raw_pixels_from_json(GT_PATH, is_coco=False)
            mediapipe_df = pd.DataFrame(load_json(MP_PATH))
            sapiens_df = pd.DataFrame(load_json(SP_PATH))
            yolo_df = pd.DataFrame(load_json(YO_PATH))

            print("\nDo you want to apply acrobatic-only evaluation? (y/-)")
            if input("> ").lower() == 'y':
                h = 720 #hardcoded from video settings
                thr_input = input("Put the coeficient (example: 0.5) [Default = 0.5]: ") or "0.5"
                thr = float(thr_input)
                y_thr = h * thr
                y_pelvis = gt_raw['y_24']
                mask = y_pelvis.notna() & (y_pelvis < y_thr)
                
                gt_raw = gt_raw[mask].reset_index(drop=True)
                mediapipe_df = mediapipe_df[mask].reset_index(drop=True)
                sapiens_df = sapiens_df[mask].reset_index(drop=True)
                yolo_df = yolo_df[mask].reset_index(drop=True)

            gt = normalize_dataframe(gt_raw)
            mediapipe = mediapipe_df
            sapiens = sapiens_df
            yolo = yolo_df

    print("\nChoose the metric you want to use:")
    print("1: mAP")
    print("2: PCK")
    m_opt = input("> ")
    metric_name = "mAP" if m_opt == "1" else "PCK"
    default_thr = "0.5" if m_opt == "1" else "0.2"
    thr = float(input(f"Select a threshold [Default value = {default_thr}]: ") or default_thr)
    
    if metric_name == "mAP":
        res_mp = map(gt, mediapipe, thr)
        res_sapiens = map(gt, sapiens, thr)
        res_yolo = map(gt, yolo, thr)
        m_key = "MEAN_mAP"
    else:
        res_mp = pck(gt, mediapipe, thr)
        res_sapiens = pck(gt, sapiens, thr)
        res_yolo = pck(gt, yolo, thr)
        m_key = "MEAN_PCK"

    print("\n" + "-" * 70)
    print(f"{'Keypoint':<25} | {'Mediapipe BlazePose':<10} | {'Sapiens@2B %':<12} | {'YOLO26x-pose %':<10}")
    print("-" * 70)
    for mp_idx in coco2mp.values():
        k = f'Point_{mp_idx}'
        print(f"Keypoint_{mp_idx:<16} | {res_mp[k]*100:>8.1f}% | {res_sapiens[k]*100:>10.1f}% | {res_yolo[k]*100:>8.1f}%")
    print("-" * 70)
    print(f"{'Mean Average (' + metric_name + ')':<25} | {res_mp[m_key]*100:>8.1f}% | {res_sapiens[m_key]*100:>10.1f}% | {res_yolo[m_key]*100:>8.1f}%")
    print("-" * 70 + "\n")