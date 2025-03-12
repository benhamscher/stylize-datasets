import subprocess
import datetime
import time
from pathlib import Path

# Base paths - using ~ for home directory
BASE_DIR = "~/datasets/Cityscapes"
CONTENT_DIRS = [f"{BASE_DIR}/leftImg8bit/train", f"{BASE_DIR}/leftImg8bit/val"]
MASK_DIRS = [f"{BASE_DIR}/gtFine/train", f"{BASE_DIR}/gtFine/val"]
STYLE_DIR = "~/datasets/train"  # Using default from your script

# Parameter combinations
NUM_POINTS = [8, 16, 32, 50]  # Different numbers of Voronoi cells
STYLIZE_PROPS = [0.75, 1.0]   # Different proportions to stylize
ALPHAS = [0.5, 0.75, 1.0]     # Different style strength values

# Generate all parameter combinations
PARAMS = []
for np in NUM_POINTS:
    for sp in STYLIZE_PROPS:
        for alpha in ALPHAS:
            # Create output paths following naming scheme
            output_dirs = [f"{BASE_DIR}/stylized/train_Voronoi{np}_prop{sp}_alpha{alpha}", 
                          f"{BASE_DIR}/stylized/val_Voronoi{np}_prop{sp}_alpha{alpha}"]
            
            # Convert lists to space-separated strings for command line
            content_dirs_str = " ".join([str(Path(p).expanduser()) for p in CONTENT_DIRS])
            mask_dirs_str = " ".join([str(Path(p).expanduser()) for p in MASK_DIRS])
            output_dirs_str = " ".join(output_dirs)
            
            param_str = (f"--num_points {np} "
                        f"--output_dirs {output_dirs_str} "
                        f"--content_dirs {content_dirs_str} "
                        f"--mask_dirs {mask_dirs_str} "
                        f"--style_dir {STYLE_DIR} "
                        f"--stylize_proportion {sp} "
                        f"--alpha {alpha}")

            PARAMS.append(param_str)

SCRIPT_PATH = "~/Masterthesis/stylize-datasets/voronoi_style_transfer_cs.py"

def run_style_transfers():
    for i, params in enumerate(PARAMS, 1):
        print(f"\n[{datetime.datetime.now()}] Starting run {i}/{len(PARAMS)}")
        print(f"Parameters: {params}")
        
        try:
            cmd = f"python {SCRIPT_PATH} {params}"
            subprocess.run(cmd, shell=True, check=True)
            
            print(f"[{datetime.datetime.now()}] Completed run {i}/{len(PARAMS)}")
            
            # Add small delay between runs
            if i < len(PARAMS):
                time.sleep(5)
                
        except subprocess.CalledProcessError as e:
            print(f"Error in run {i}: {e}")
            continue

if __name__ == "__main__":
    print(f"Will run {len(PARAMS)} parameter combinations")
    run_style_transfers()