import torch
from PIL import Image
import numpy as np
from pathlib import Path
import random
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from tqdm import tqdm
from stylize import style_transfer, input_transform
import net
from labels import trainId2label

def find_image_pairs(rgb_dir, mask_dir, pattern='*_*_leftImg8bit.png', panoptic_mask_suffix='_gtFine_panoptic.png', semantic_mask_suffix='_gtFine_labelTrainIds.png'):
    """
    Finds and pairs RGB images with their corresponding panoptic and semantic masks.
    """
    rgb_paths = list(Path(rgb_dir).rglob(pattern))
    image_pairs = {}

    for rgb_path in rgb_paths:
        base_name = rgb_path.stem.replace('_leftImg8bit', '')
        city_name = rgb_path.parent.name
        panoptic_mask_filename = f"{base_name}{panoptic_mask_suffix}"
        semantic_mask_filename = f"{base_name}{semantic_mask_suffix}"
        panoptic_mask_path = Path(mask_dir) / city_name / panoptic_mask_filename
        semantic_mask_path = Path(mask_dir) / city_name / semantic_mask_filename

        if panoptic_mask_path.exists() and semantic_mask_path.exists():
            image_pairs[str(rgb_path)] = (str(panoptic_mask_path), str(semantic_mask_path))
        else:
            print(f"Warning: Mask path for {rgb_path} does not exist.")

    return image_pairs

def stylize_panoptic(content, style, vgg, decoder, alpha, device):
    """
    Perform style transfer on a single panoptic segment.
    """
    with torch.no_grad():
        content = content.to(device).unsqueeze(0)
        style = style.to(device).unsqueeze(0)
        output = style_transfer(vgg, decoder, content, style, alpha)
    return output

def get_semantic_class(semantic_mask, panoptic_id, panoptic_ids):
    """
    Get the semantic class for a given panoptic ID using its location in the panoptic mask.
    """
    locations = np.where(panoptic_ids == panoptic_id)
    if len(locations[0]) > 0:
        # Get the first pixel location of this panoptic ID
        y, x = locations[0][0], locations[1][0]
        return semantic_mask[y, x]
    return None

def collect_style_images(style_dir, split):
    """
    Collect style images organized by class ID in the new folder structure.
    """
    styles = {}
    base_dir = Path(style_dir)
    for class_dir in base_dir.iterdir():
        # if class_dir.is_dir() and class_dir.name.startswith("2023_03_03_upsampled_texture_images_in_contour_"):
        if class_dir.is_dir() and class_dir.name.startswith("trainID"):
            class_id = class_dir.name.split('_')[-1]
            styles[class_id] = []
            # split_dir = class_dir / split
            split_dir = class_dir
            for img_path in split_dir.rglob('*.png'):
                styles[class_id].append(str(img_path))
    return styles

def select_style(semantic_class, styles):
    """
    Select a style from a different class than the semantic_class.
    """
    different_classes = [cls for cls in styles.keys() if cls != str(semantic_class)]
    if not different_classes:
        return random.choice([style for class_styles in styles.values() for style in class_styles])
    selected_class = random.choice(different_classes)
    return random.choice(styles[selected_class])

def main():
    content_dirs = ['/home/bhamscher/datasets/Cityscapes/leftImg8bit/train', 
                    '/home/bhamscher/datasets/Cityscapes/leftImg8bit/val']
    style_dir = '/home/bhamscher/results/patches_greater_100'
    output_dirs = ['/home/bhamscher/datasets/Cityscapes_cue_conflict_stylized_patches_greater_100/leftImg8bit_cue_conflict/train', 
                   '/home/bhamscher/datasets/Cityscapes_cue_conflict_stylized_patches_greater_100/leftImg8bit_cue_conflict/val']
    mask_dirs = ['/home/bhamscher/datasets/Cityscapes/gtFine/train', 
                 '/home/bhamscher/datasets/Cityscapes/gtFine/val']
    alpha = 1.0
    content_size = 0
    style_size = 512
    crop = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stylize_proportion = 1.0

    for content_dir, output_dir, mask_dir in zip(content_dirs, output_dirs, mask_dirs):
        print(f"Stylizing Images in {content_dir}.")
        content_dir = Path(content_dir).resolve()
        style_dir = Path(style_dir).resolve()
        output_dir = Path(output_dir).resolve()
        mask_dir = Path(mask_dir).resolve()
        split = content_dir.parts[-1]

        # Prepare model
        decoder = net.decoder
        vgg = net.vgg

        decoder.eval()
        vgg.eval()

        decoder.load_state_dict(torch.load('models/decoder.pth'))
        vgg.load_state_dict(torch.load('models/vgg_normalised.pth'))
        vgg = nn.Sequential(*list(vgg.children())[:31])

        vgg.to(device)
        decoder.to(device)

        # Find image pairs
        image_pairs = find_image_pairs(content_dir, mask_dir)
        print(f'Found {len(image_pairs)} image pairs')

        # Collect style files and organize by class ID
        styles = collect_style_images(style_dir, split=split)
        print(f'Found styles for {len(styles)} classes in {style_dir}')

        # Disable decompression bomb errors
        Image.MAX_IMAGE_PIXELS = None
        skipped_imgs = []

        # Define transforms using input_transform function
        content_tf = input_transform(content_size, crop)
        style_tf = input_transform(style_size, 0)

        # Process images
        num_images = len(image_pairs)
        # num_images = 2
        with tqdm(total=num_images) as pbar:
            for (content_path, (panoptic_mask_path, semantic_mask_path)) in list(image_pairs.items())[:num_images]:
                try:
                    # Load content image and masks
                    content_img = Image.open(content_path).convert('RGB')
                    content_tensor = content_tf(content_img)

                    panoptic = np.array(Image.open(panoptic_mask_path))
                    semantic = np.array(Image.open(semantic_mask_path))

                    # COCO dataset panoptic IDs
                    panoptic_ids = panoptic[:,:,0] + panoptic[:,:,1] * 256 + panoptic[:,:,2] * 256**2
                    unique_ids = np.unique(panoptic_ids)

                    # Only choose proportion of IDs to stylize
                    if stylize_proportion != 1.0:
                        unique_ids = np.random.choice(unique_ids, int(stylize_proportion * len(unique_ids)), replace=False)

                    # Initialize result tensor and new mask for style IDs
                    result = torch.zeros_like(content_tensor).unsqueeze(0).to(device)
                    new_style_mask = np.zeros_like(panoptic_ids)  # Initialize the new mask for style classes

                    # Process each unique ID
                    for id in unique_ids:
                        # Get semantic class for this panoptic ID
                        semantic_class = get_semantic_class(semantic, id, panoptic_ids)
                        
                        if semantic_class is not None:
                            # Select style from a different class
                            style_path = select_style(semantic_class, styles)

                            # Load and preprocess style image
                            style_img = Image.open(style_path).convert('RGB')
                            style_tensor = style_tf(style_img)

                            # Perform style transfer
                            stylized_output = stylize_panoptic(content_tensor, style_tensor, vgg, decoder, alpha, device)

                            # Create and apply mask
                            id_mask = torch.from_numpy((panoptic_ids == id).astype(np.float32))
                            mask_3d = id_mask.unsqueeze(0).repeat(3, 1, 1).to(device)
                            result += mask_3d * stylized_output

                            # Assign the selected style class ID to the new mask
                            # selected_style_class = style_path.split('/')[-3].split('_')[-1]  # Extract class ID from style path
                            selected_style_class = style_path.split('/')[-2].split('_')[-1]  # Extract class ID from style path
                            new_style_mask[panoptic_ids == id] = selected_style_class

                    # Save result with maintained directory structure
                    content_path = Path(content_path)
                    rel_path = content_path.relative_to(content_dir)
                    out_dir = output_dir.joinpath(rel_path.parent)

                    # Create directory structure if it does not exist
                    out_dir.mkdir(parents=True, exist_ok=True)

                    content_name = content_path.stem
                    out_filename = f"{content_name}{content_path.suffix}"
                    output_name = out_dir.joinpath(out_filename)

                    # Save the stylized image
                    save_image(result.cpu(), output_name, padding=0)

                    # Save the new mask as a PNG
                    new_mask_name = f"{content_name}_style_mask.png"
                    new_mask_path = out_dir.joinpath(new_mask_name)
                    Image.fromarray(new_style_mask.astype(np.uint8)).save(new_mask_path)

                    # Create and save the RGB style mask
                    height, width = new_style_mask.shape
                    rgb_array = np.zeros((height, width, 3), dtype=np.uint8)

                    # Map train IDs to RGB colors
                    for train_id in range(19):
                        rgb_color = trainId2label[train_id].color
                        rgb_array[new_style_mask == train_id] = rgb_color

                    # Save the RGB image
                    rgb_image = Image.fromarray(rgb_array)
                    rgb_image_name = f"{content_name}_style_mask_rgb.png"
                    rgb_image_path = out_dir.joinpath(rgb_image_name)
                    rgb_image.save(rgb_image_path)

                    print(f"Image processed and saved to {output_name}")
                    print(f"New style mask saved to {new_mask_path}")
                    print(f"RGB style mask saved to {rgb_image_path}")

                except Exception as e:
                    print(f'Skipping stylization of {content_path} due to an error: {str(e)}')
                    skipped_imgs.append(content_path)
                    continue
                finally:
                    pbar.update(1)

        if skipped_imgs:
            with open(output_dir / 'skipped_imgs.txt', 'w') as f:
                for item in skipped_imgs:
                    f.write(f"{item}\n")

if __name__ == '__main__':
    main()