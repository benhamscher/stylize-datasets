import torch
from PIL import Image
import numpy as np
import os
import glob
import numpy as np
from scipy.ndimage import label
import argparse
from function import adaptive_instance_normalization
import net
from pathlib import Path
import random
import torch
import torch.nn as nn
import torchvision.transforms
from torchvision.utils import save_image
from tqdm import tqdm
from stylize import input_transform, style_transfer

def extract_bounding_boxes(panoptic_mask_path, rgb_image_path, device):
    """
    Extracts bounding boxes from an RGB image based on the panoptic anno    	tation.
    
    Args:
        panoptic_mask_path (str): Path to the panoptic mask image.
        rgb_image_path (str): Path to the RGB image.

    Returns:
        dict: A dictionary where keys are unique IDs and values are tuples containing
              bounding box (top, bottom, left, right) and the corresponding RGB patch.
    """
    
    # Load panoptic mask and RGB image
    panoptic_mask = Image.open(panoptic_mask_path)
    rgb_image = Image.open(rgb_image_path).convert('RGB')
    
    # Convert images to torch tensors and move to GPU
    panoptic_mask = torch.tensor(np.array(panoptic_mask)).to(device)
    rgb_image = torch.tensor(np.array(rgb_image)).to(device)

    # Get unique IDs from the panoptic mask
    unique_ids = torch.unique(panoptic_mask)
    bounding_boxes = {}
    
    # For each unique ID, find the bounding box and extract the corresponding region
    for obj_id in unique_ids:
        # if obj_id == 0:  # Skip background (assuming background ID is 0)
        #     continue
        
        # Create a binary mask for the current object
        obj_mask = (panoptic_mask == obj_id)
        
        # Find the coordinates for the bounding box
        coords = torch.where(obj_mask)
        rows, cols = coords[0], coords[1]  # Unpacking the tuple (rows, cols)
        
        if rows.numel() == 0 or cols.numel() == 0:
            continue  # Skip if no valid pixels found
        
        # Bounding box coordinates
        # Calculate bounding box coordinates using PyTorch functions
        # The extra pixels are to assure functionality of the net during pooling
        top = torch.clamp(torch.min(rows) - 10, min=0).item()
        bottom = torch.clamp(torch.max(rows) + 10, max=panoptic_mask.shape[0]).item()
        left = torch.clamp(torch.min(cols) - 10, min=0).item()
        right = torch.clamp(torch.max(cols) + 10, max=panoptic_mask.shape[1]).item()
        
        # Extract the bounding box from the RGB image
        rgb_patch = rgb_image[top:bottom+1, left:right+1]

        # Ensure rgb_patch is a valid tensor
        bounding_boxes[obj_id.item()] = {
                'bbox': (top, bottom, left, right),
                'rgb_patch': rgb_patch,
                'img_path': Path(rgb_image_path)
            }

    return rgb_image, panoptic_mask, bounding_boxes



def find_image_pairs(rgb_dir, mask_dir, pattern='*_*_leftImg8bit.png', mask_suffix='_gtFine_panoptic.png'):
    """
    Finds and pairs RGB images with their corresponding panoptic masks.

    Args:
        rgb_dir (str): Directory containing RGB images.
        mask_dir (str): Directory containing panoptic masks.
        pattern (str): Pattern to match RGB images.
        mask_suffix (str): Suffix to identify corresponding panoptic masks.

    Returns:
        dict: A dictionary where keys are RGB image paths and values are corresponding panoptic mask paths.
    """
    rgb_paths = glob.glob(os.path.join(rgb_dir, pattern))
    image_pairs = {}

    for rgb_path in rgb_paths:
        # Extract the base filename (without suffix)
        base_name = os.path.basename(rgb_path).replace('_leftImg8bit.png', '')

        # Construct the corresponding panoptic mask path
        mask_filename = base_name + mask_suffix
        mask_path = os.path.join(mask_dir, mask_filename)

        # Check if the mask file exists
        if os.path.exists(mask_path):
            image_pairs[rgb_path] = mask_path

    return image_pairs

def stylize_boxes(boxes, style_dir, content_tf, style_tf, device, vgg, decoder, alpha):
    assert style_dir.is_dir(), 'Style directory not found'
    extensions = ['png', 'jpeg', 'jpg']
    styles = []
    for ext in extensions:
        styles += list(style_dir.rglob('*.' + ext))
    assert len(styles) > 0, 'No images with specified extensions found in style directory' + style_dir
    styles = sorted(styles)

    stylized_boxes = {}
    style_paths = random.sample(styles, len(boxes.keys()))
    for idx, key in enumerate(boxes.keys()):
        box = boxes[key]
        content_img = box["rgb_patch"]
        style_path = style_paths[idx]
        style_img = None

        while style_img is None:
            try:
                style_img = Image.open(style_path).convert('RGB')
            except Exception as e:
                print(f'Error loading style image {style_path}: {e}')
                # Select a new random style image if loading fails
                style_path = random.choice(styles)
                # Skip to the next iteration if there's an error

        style_img = Image.open(style_path).convert('RGB')
        content = content_img.permute(2, 0, 1).float()
        print("Content:", content.shape)
        style = style_tf(style_img)
        style = style.to(device).unsqueeze(0)
        content = content.to(device).unsqueeze(0)
            
        with torch.no_grad():
            output = style_transfer(vgg, decoder, content, style, alpha)
        print("stylized_result:", output.shape)
        stylized_boxes[key] = {
                'bbox': box['bbox'],
                'stylized_img': output,
                'img_path': box['img_path']
            }
    return stylized_boxes
            
def patchwork(rgb_tensor, panoptic_mask, stylized_boxes, output_dir):
    # Create a copy of the original image tensor to modify
    result_img_tensor = rgb_tensor.clone()
    print("result_img:", result_img_tensor.shape)
    for key, value in stylized_boxes.items():
        bbox = value['bbox']
        stylized_img = value['stylized_img'].squeeze().permute(1,2,0).int()
        print("Stylized:", stylized_img.shape)
        top, bottom, left, right = bbox
        print("tblr:", top, bottom, left, right)
        obj_id = torch.tensor(key)
        obj_mask = (panoptic_mask == obj_id)
        mask_area = obj_mask[top:bottom, left:right]
        print("Mask area:", mask_area.shape)
        # Apply the mask to the stylized image and the original image tensor
        result_img_tensor[top:bottom, left:right, :] = (mask_area[:, :] * stylized_img)
    result_img = Image.fromarray(result_img_tensor.cpu().detach().numpy())
    img_name = stylized_boxes[stylized_boxes.keys[0]["img_path"]].name
    result_img.save(os.path.join(output_dir, img_name))

        


def main():
    device = "cuda:0"
    rgb_img, panoptic_mask, bboxes = extract_bounding_boxes("/home/bhamscher/datasets/Cityscapes/gtFine/train/aachen/aachen_000000_000019_gtFine_panoptic.png", 
                                     "/home/bhamscher/datasets/Cityscapes/leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png", device)
    # bboxes = extract_segments("/home/bhamscher/datasets/Cityscapes/gtFine/train/aachen/aachen_000000_000019_gtFine_panoptic.png", 
    #                                  "/home/bhamscher/datasets/Cityscapes/leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png")
    # for idx in bboxes.keys():
    #     box = bboxes[idx]
    #     patch = box["rgb_patch"].cpu().detach().numpy()
    #     patch_im = Image.fromarray(patch)
    #     patch_im.save(f"/home/bhamscher/results/bbox_test/box_{idx}.png")
    
    

    content_size = 0
    crop = 0
    style_size = 512
    alpha = 1
    style_dir = "/home/bhamscher/datasets/train"
    style_dir = Path(style_dir).resolve()
    # output_dir = Path(output_dir).resolve()
    
    # initialize nets and load weights
    decoder = net.decoder
    vgg = net.vgg

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    decoder.eval()
    vgg.eval()

    decoder.load_state_dict(torch.load('/home/bhamscher/Masterthesis/stylize-datasets/models/decoder.pth'))
    vgg.load_state_dict(torch.load('/home/bhamscher/Masterthesis/stylize-datasets/models/vgg_normalised.pth'))
    vgg = nn.Sequential(*list(vgg.children())[:31])

    vgg.to(device)
    decoder.to(device)

    content_tf = input_transform(content_size, crop)
    style_tf = input_transform(style_size, 0)
    stylized_boxes = stylize_boxes(bboxes, style_dir, content_tf, style_tf, device, vgg, decoder, alpha)
    # for idx in stylized_boxes.keys():
    #     box = stylized_boxes[idx]
    #     patch = box["stylized_img"].cpu()
    #     print(patch.shape)
    #     save_image(patch, f"/home/bhamscher/results/bbox_test/stylized_box_{idx}.png", padding=0) #default image padding is 2.

    patchwork(rgb_img, panoptic_mask,stylized_boxes, "/home/bhamscher/results/bbox_test")

if __name__ == "__main__":
        main()