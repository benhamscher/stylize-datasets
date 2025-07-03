import subprocess
import datetime
import time
from pathlib import Path

# Base paths
BASE_DIR = "~/datasets/PASCALVOC/VOCdevkit/VOC2010"
CONTENT_DIR = f"{BASE_DIR}/JPEGImages_3_split_cropped"
MASK_DIR = f"{BASE_DIR}/SemsegIDImg_3_split_cropped"

# Parameter combinations
NUM_POINTS = [4, 8, 16] # adjust number of ovornoi cells as needed
STYLIZE_PROPS = [0.25, 0.5, 0.75, 1.0] # adjust stylize proportion as needed
ALPHAS =  [1.0] # [0.25, 0.5, 0.75, 1.0] # adjust alpha as needed

# Generate all parameter combinations
PARAMS = []
for np in NUM_POINTS:
    for sp in STYLIZE_PROPS:
        for alpha in ALPHAS:
            # Create output path following naming scheme
            output_base = f"{BASE_DIR}/JPEGImages_3_split_stylized_Voronoi{np}_stylize_prop{sp}_alpha{alpha}_corrected"
            
            param_str = (f"--num_points {np} "
                        f"--output_dirs {output_base} "
                        f"--content_dirs {CONTENT_DIR} "
                        f"--mask_dirs {MASK_DIR} "
                        f"--stylize_proportion {sp} "
                        f"--alpha {alpha} "
                        f"--txt_path '{output_base}/skipped_imgs.txt'" # txt_path optional for redo of missed images, comment out
                        ) 

            PARAMS.append(param_str)

SCRIPT_PATH = "~/stylize-datasets/voronoi_style_transfer_pc.py" 

def run_style_transfers():
    for i, params in enumerate(PARAMS, 1):
        print(f"\n[{datetime.datetime.now()}] Starting run {i}/{len(PARAMS)}")
        print(f"Parameters: {params}")
        
        try:
            cmd = f"python {SCRIPT_PATH} {params}"
            subprocess.run(cmd, shell=True, check=True)
            
            print(f"[{datetime.datetime.now()}] Completed run {i}")
            
            # Add small delay between runs
            if i < len(PARAMS):
                time.sleep(5)
                
        except subprocess.CalledProcessError as e:
            print(f"Error in run {i}: {e}")
            continue

if __name__ == "__main__":
    run_style_transfers()