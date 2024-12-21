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

def find_image_pairs(rgb_dir, mask_dir, pattern='*_*_leftImg8bit.png', mask_suffix='_gtFine_panoptic.png'):
    """
    Finds and pairs RGB images with their corresponding panoptic masks.
    """
    rgb_paths = sorted(list(Path(rgb_dir).rglob(pattern)))
    mask_paths = list(Path(mask_dir).rglob(f"*{mask_suffix}"))
    random.shuffle(mask_paths)
    image_pairs = {}

    for rgb_path in rgb_paths:
        mask_path = mask_paths.pop()
        image_pairs[str(rgb_path)] = str(mask_path)

    # for rgb_path in rgb_paths:
    #     base_name = rgb_path.stem.replace('_leftImg8bit', '')
    #     city_name = rgb_path.parent.name
    #     mask_filename = f"{base_name}{mask_suffix}"
    #     mask_path = Path(mask_dir) / city_name / mask_filename

    #     if mask_path.exists():
    #         image_pairs[str(rgb_path)] = str(mask_path)
    #     else:
    #         print(f"Warning: Mask path for {rgb_path} does not exist.")

    return image_pairs

def stylize_panoptic(content, style, vgg, decoder, alpha, device):
    """
    Perform style transfer on a single panoptic segment.
    """
    with torch.no_grad():
        content = content.to(device).unsqueeze(0)
        style = style.to(device).unsqueeze(0)
        output = style_transfer(vgg, decoder, content, style, alpha)
    return output#.detach().cpu().squeeze(0)


def main():
    # Define arguments here instead of using argparse
    content_dirs = ['/home/bhamscher/datasets/Cityscapes/leftImg8bit/train', 
                    '/home/bhamscher/datasets/Cityscapes/leftImg8bit/val']
    style_dir = '/home/bhamscher/datasets/train'

    # Adjust output dirs according to wishes
    output_dirs = ['/home/bhamscher/datasets/Cityscapes_stylized_panoptic_wrong_mask/leftImg8bit_stylized_panoptic/train', 
                   '/home/bhamscher/datasets/Cityscapes_stylized_panoptic_wrong_mask/leftImg8bit_stylized_panoptic/val']
    mask_dirs = ['/home/bhamscher/datasets/Cityscapes/gtFine/train', 
                 '/home/bhamscher/datasets/Cityscapes/gtFine/val']
    extensions = ['png', 'jpeg', 'jpg']
    alpha = 1.0
    content_size = 0
    style_size = 512
    crop = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    stylize_proportion = 1.0 # Set the proportion of IDs to stylize, 1.0 for full stylization
    assert stylize_proportion <= 1.0 and stylize_proportion >= 0.0, "Invalid proportion of image to stylize."

    for content_dir, output_dir, mask_dir in zip(content_dirs, output_dirs, mask_dirs):
        print(f"Stylizing Images in {content_dir}.")
        content_dir = Path(content_dir).resolve()
        style_dir = Path(style_dir).resolve()
        output_dir = Path(output_dir).resolve()
        mask_dir = Path(mask_dir).resolve()

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

        # collect style files
        styles = []
        for ext in extensions:
            styles += list(style_dir.rglob('*.' + ext))

        assert len(styles) > 0, 'No images with specified extensions found in style directory' + str(style_dir)
        styles = sorted(styles)
        print('Found %d style images in %s' % (len(styles), style_dir))

        # Disable decompression bomb errors
        Image.MAX_IMAGE_PIXELS = None
        skipped_imgs = []

        # Define transforms using input_transform function
        content_tf = input_transform(content_size, crop)
        style_tf = input_transform(style_size, 0)

        # Process images
        # num_images = min(5, len(image_pairs))  # Process first 5 images or all if less than 5
        num_images = len(image_pairs)
        num_images = 10
        with tqdm(total=num_images) as pbar:
            for (content_path, mask_path) in list(image_pairs.items())[:num_images]:
                try:
                    # Load content image and mask
                    content_img = Image.open(content_path).convert('RGB')
                    content_tensor = content_tf(content_img)

                    panoptic = np.array(Image.open(mask_path))

                    # COCO dataset panoptic IDs
                    panoptic_ids = panoptic[:,:,0] + panoptic[:,:,1] * 256 + panoptic[:,:,0] * 256**2
                    unique_ids = np.unique(panoptic_ids)

                    # Only choose proportion of IDs to stylize
                    if stylize_proportion != 1.0:
                        unique_ids = np.random.choice(unique_ids, int(stylize_proportion * len(unique_ids)), replace = False)

                    # Select styles
                    selected_styles = random.sample(styles, min(len(unique_ids), len(styles)))

                    # Initialize result tensor, starting from original image if stylize_proportion != 1.0
                    if stylize_proportion != 1.0:
                        result = content_tensor.unsqueeze(0).to(device)
                    else:
                        result = torch.zeros_like(content_tensor).unsqueeze(0).to(device)

                    # Process each unique ID
                    for id, style_path in zip(unique_ids, selected_styles):
                        # Load and preprocess style image
                        style_img = Image.open(style_path).convert('RGB')
                        style_tensor = style_tf(style_img)

                        # Perform style transfer
                        stylized_output = stylize_panoptic(content_tensor, style_tensor, vgg, decoder, alpha, device)

                        # Create and apply mask
                        id_mask = torch.from_numpy((panoptic_ids == id).astype(np.float32))
                        mask_3d = id_mask.unsqueeze(0).repeat(3, 1, 1).to(device)
                        result += mask_3d * stylized_output

                    # Save result with maintained directory structure
                    content_path = Path(content_path)
                    rel_path = content_path.relative_to(content_dir)
                    out_dir = output_dir.joinpath(rel_path.parent)

                    # Create directory structure if it does not exist
                    out_dir.mkdir(parents=True, exist_ok=True)

                    content_name = content_path.stem
                    out_filename = f"{content_name}{content_path.suffix}" # Without"_stylized" to simplify data loading
                    output_name = out_dir.joinpath(out_filename)

                    save_image(result.cpu(), output_name, padding=0)

                    print(f"Image processed and saved to {output_name}")

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

def process_single_image(content_path, mask_path, style_dir, output_path, alpha=1.0, content_size=0, style_size=512, crop=0):
    # 1. Load content image as tensor
    content_img = Image.open(content_path).convert('RGB')
    content_tf = input_transform(content_size, crop)
    content_tensor = content_tf(content_img)

    # 2. Load panoptic mask as numpy array and identify unique IDs in the red channel
    panoptic = np.array(Image.open(mask_path))
    panoptic_ids = panoptic[:,:,0] + panoptic[:,:,1] * 256 + panoptic[:,:,0] * 256**2
    unique_ids = np.unique(panoptic_ids)
    print(f"Number of unique IDs: {len(unique_ids)}")
    print("IDs", unique_ids)

    # 3. Select styles based on number of unique IDs
    style_paths = list(Path(style_dir).rglob('*.*'))
    print(f"Found {len(style_paths)} style images in {style_dir}")
    selected_styles = random.sample(style_paths, min(len(unique_ids), len(style_paths)))

    # Prepare model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder = net.decoder
    vgg = net.vgg

    decoder.eval()
    vgg.eval()

    decoder.load_state_dict(torch.load('models/decoder.pth'))
    vgg.load_state_dict(torch.load('models/vgg_normalised.pth'))
    vgg = nn.Sequential(*list(vgg.children())[:31])

    vgg.to(device)
    decoder.to(device)

    # 4. Perform stylization for each ID and add to list
    stylized_images = []
    style_tf = input_transform(style_size, 0)
    for id, style_path in zip(unique_ids, selected_styles):
        # Load and preprocess style image
        style_img = Image.open(style_path).convert('RGB')
        style_tensor = style_tf(style_img)

        # Perform style transfer on the content
        stylized_output = stylize_panoptic(content_tensor, style_tensor, vgg, decoder, alpha, device)
        
        stylized_images.append((id, stylized_output))

    # 5. Combine stylized images
    result = torch.zeros_like(content_tensor).unsqueeze(0).to(device)
    for id, stylized_image in stylized_images:
        # 5.1 Create 3D binary mask
        id_mask = torch.from_numpy((panoptic_ids == id).astype(np.float32))
        mask_3d = id_mask.unsqueeze(0).repeat(3, 1, 1).to(device)  # Expand to 3 channels
        
        # 5.2 Add masked stylized image to result
        result += mask_3d * stylized_image

    # 6. Save result
    result = result.cpu()
    save_image(result, output_path, padding=0)

    print(f"Image processed and saved to {output_path}")

if __name__ == '__main__':
    # style_dir = '/home/bhamscher/datasets/train'
    # content_paths = ['/home/bhamscher/datasets/Cityscapes/leftImg8bit/val/frankfurt/frankfurt_000000_000576_leftImg8bit.png',
    #                 '/home/bhamscher/datasets/Cityscapes/leftImg8bit/val/frankfurt/frankfurt_000000_011007_leftImg8bit.png',
    #                 '/home/bhamscher/datasets/Cityscapes/leftImg8bit/val/frankfurt/frankfurt_000000_020880_leftImg8bit.png']
    # mask_paths = ['/home/bhamscher/datasets/Cityscapes/gtFine/val/frankfurt/frankfurt_000000_000576_gtFine_panoptic.png',
    #                 '/home/bhamscher/datasets/Cityscapes/gtFine/val/frankfurt/frankfurt_000000_011007_gtFine_panoptic.png',
    #                 '/home/bhamscher/datasets/Cityscapes/gtFine/val/frankfurt/frankfurt_000000_020880_gtFine_panoptic.png']
                    
    # output_paths = ['/home/bhamscher/datasets/Cityscapes/leftImg8bit_stylized_panoptic/val/frankfurt/frankfurt_000000_000576_leftImg8bit.png',
    #                 '/home/bhamscher/datasets/Cityscapes/leftImg8bit_stylized_panoptic/val/frankfurt/frankfurt_000000_011007_leftImg8bit.png',
    #                 '/home/bhamscher/datasets/Cityscapes/leftImg8bit_stylized_panoptic/val/frankfurt/frankfurt_000000_020880_leftImg8bit.png']
                    
    # for cp, mp, op in zip(content_paths, mask_paths, output_paths):
    #     process_single_image(cp, mp, style_dir, op)
    main()