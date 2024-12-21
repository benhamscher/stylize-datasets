import torch
from PIL import Image
import numpy as np
import os
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
    rgb_paths = list(Path(rgb_dir).rglob(pattern))
    image_pairs = {}

    for rgb_path in rgb_paths:
        base_name = rgb_path.stem.replace('_leftImg8bit', '')
        mask_filename = f"{base_name}{mask_suffix}"
        mask_path = Path(mask_dir) / mask_filename

        if mask_path.exists():
            image_pairs[str(rgb_path)] = str(mask_path)

    return image_pairs

def stylize_panoptic(content, style, vgg, decoder, alpha, device):
    """
    Perform style transfer on a single panoptic segment.
    """
    with torch.no_grad():
        content = content.to(device).unsqueeze(0)
        style = style.to(device).unsqueeze(0)
        output = style_transfer(vgg, decoder, content, style, alpha)
    return output.detach().cpu().squeeze(0)

def main():
    # Define arguments here instead of using argparse
    content_dir = 'path/to/content/dir'
    style_dir = 'path/to/style/dir'
    output_dir = 'path/to/output/dir'
    mask_dir = 'path/to/mask/dir'
    alpha = 1.0
    num_styles = 1

    content_dir = Path(content_dir)
    style_dir = Path(style_dir)
    output_dir = Path(output_dir)
    mask_dir = Path(mask_dir)

    # Find image pairs
    image_pairs = find_image_pairs(content_dir, mask_dir)

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

    # Collect style files
    styles = list(style_dir.rglob('*.*'))
    print(f'Found {len(styles)} style images in {style_dir}')

    # Disable decompression bomb errors
    Image.MAX_IMAGE_PIXELS = None
    skipped_imgs = []

    with tqdm(total=len(image_pairs)) as pbar:
        for content_path, mask_path in image_pairs.items():
            try:
                content_img = Image.open(content_path).convert('RGB')
                mask_img = Image.open(mask_path)
                mask_array = np.array(mask_img)

                # Get unique IDs from the mask
                unique_ids = np.unique(mask_array)

                for style_path in random.sample(styles, num_styles):
                    style_img = Image.open(style_path).convert('RGB')
                    
                    # Resize style image to match content image size
                    style_img = style_img.resize(content_img.size, Image.LANCZOS)
                    
                    # Convert images to tensors
                    content_tensor = transforms.ToTensor()(content_img)
                    style_tensor = transforms.ToTensor()(style_img)

                    # Initialize the final output image
                    final_output = torch.zeros_like(content_tensor)

                    for id in unique_ids:
                        # Create a binary mask for the current ID
                        id_mask = (mask_array == id).astype(np.float32)
                        id_mask_tensor = torch.from_numpy(id_mask).unsqueeze(0)

                        # Apply the mask to the content image
                        masked_content = content_tensor * id_mask_tensor

                        # Perform style transfer on the masked content
                        stylized_output = stylize_panoptic(masked_content, style_tensor, vgg, decoder, alpha, device)

                        # Add the stylized output to the final image using the binary mask
                        final_output += stylized_output * id_mask_tensor

                    # Save the final stylized image
                    output_path = output_dir / f"{Path(content_path).stem}_stylized_{Path(style_path).stem}.png"
                    save_image(final_output, output_path, padding=0)

                    style_img.close()
                content_img.close()
                mask_img.close()

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
    panoptic_ids = panoptic[:,:,0] + panoptic[:,:,1] * 256 + panoptic[:,:,0] * 256^2
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
    result = np.zeros_like(panoptic, dtype=np.float32)
    for id, stylized_image in stylized_images:
        # 5.1 Convert [C,H,W] tensor to [H,W,C] numpy array
        stylized_array = stylized_image.permute(1, 2, 0).numpy()
        # stylized_array = (stylized_array * 255).astype('uint8')  # Scale to 0-255 and convert to uint8
        # stylized_path = os.path.join(os.path.dirname(output_path), f"{id}.png")
        # # Image.fromarray(stylized_array).save(style_path)
        # save_image(stylized_image, stylized_path, padding=0) #default image padding is 2.
        
        # 5.2 Create 3D binary mask
        mask = (panoptic_ids == id)
        mask_3d = np.repeat(mask[:,:,np.newaxis], 3, axis=2)
        
        # 5.3 Add masked stylized image to result
        result += mask_3d * stylized_array

    # 6. Save result
    result = (result * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(result).save(output_path)

    print(f"Image processed and saved to {output_path}")

if __name__ == '__main__':
    # content_paths = ['/home/bhamscher/datasets/Cityscapes/leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png',
    #                  '/home/bhamscher/datasets/Cityscapes/leftImg8bit/train/bochum/bochum_000000_000313_leftImg8bit.png',
    #                  '/home/bhamscher/datasets/Cityscapes/leftImg8bit/train/bochum/bochum_000000_000600_leftImg8bit.png',
    #                  '/home/bhamscher/datasets/Cityscapes/leftImg8bit/train/bochum/bochum_000000_000885_leftImg8bit.png',
    #                  '/home/bhamscher/datasets/Cityscapes/leftImg8bit/train/bochum/bochum_000000_001097_leftImg8bit.png']
    # mask_paths = ['/home/bhamscher/datasets/Cityscapes/gtFine/train/aachen/aachen_000000_000019_gtFine_panoptic.png',
    #               '/home/bhamscher/datasets/Cityscapes/gtFine/train/bochum/bochum_000000_000313_gtFine_panoptic.png',
    #               '/home/bhamscher/datasets/Cityscapes/gtFine/train/bochum/bochum_000000_000600_gtFine_panoptic.png',
    #               '/home/bhamscher/datasets/Cityscapes/gtFine/train/bochum/bochum_000000_000885_gtFine_panoptic.png',
    #               '/home/bhamscher/datasets/Cityscapes/gtFine/train/bochum/bochum_000000_001097_gtFine_panoptic.png']
    # style_dir = '/home/bhamscher/datasets/train'
    # output_paths = [f'/home/bhamscher/results/id_style_transfer/output_image_{idx}.png' for idx in range(5)]
    content_paths = ['/home/bhamscher/datasets/Cityscapes/leftImg8bit/val/frankfurt/frankfurt_000000_000294_leftImg8bit.png']
    mask_paths = ['/home/bhamscher/datasets/Cityscapes/gtFine/val/frankfurt/frankfurt_000000_000294_gtFine_panoptic.png']
    style_dir = '/home/bhamscher/datasets/train'
    output_paths = ['/home/bhamscher/results/id_style_transfer/output_image_frankfurt_v2.png']
    for cp, mp, op in zip(content_paths,mask_paths,output_paths):
        process_single_image(cp, mp, style_dir, op)