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
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from PIL import Image, ImageDraw
import argparse

def collect_style_images(style_dir):
    """
    Collect style images organized by class ID in the new folder structure.
    """
    styles = {}
    base_dir = Path(style_dir)
    for class_dir in base_dir.iterdir():
        if class_dir.is_dir() and class_dir.name.startswith("trainID"):
            class_id = class_dir.name.split('_')[-1]
            styles[class_id] = []
            for img_path in class_dir.rglob('*.png'):
                styles[class_id].append(str(img_path))
    return styles

def voronoi_finite_polygons_2d(vor, radius=None):
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")
    new_regions = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue

        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue

            t = vor.points[p2] - vor.points[p1]
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

def generate_voronoi_mask(height, width, num_points, available_IDs):
    # Generate random points
    points = np.random.rand(num_points, 2) * [width, height]
    vor = Voronoi(points)
    regions, vertices = voronoi_finite_polygons_2d(vor)
    
    # Create mask with correct dimensions (swapped height and width)
    mask = np.zeros((height, width), dtype=np.int32)
    
    # Draw each region with a random class ID
    img = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(img)
    
    for region in regions:
        # Convert vertices to pixel coordinates
        polygon = vertices[region]
        polygon = [(int(x), int(y)) for x, y in polygon]
        # Assign a random class ID from available IDs
        class_id = int(random.choice(available_IDs))
        draw.polygon(polygon, outline=class_id, fill=class_id)
    
    # Convert to numpy array
    mask = np.array(img)
    return mask


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

# Commented out the old select_style function
# def select_style(semantic_class, styles):
#     """
#     Select a style from a different class than the semantic_class.
#     """
#     different_classes = [cls for cls in styles.keys() if cls != str(semantic_class)]
#     if not different_classes:
#         return random.choice([style for class_styles in styles.values() for style in class_styles])
#     selected_class = random.choice(different_classes)
#     return random.choice(styles[selected_class])

def main():
    parser = argparse.ArgumentParser(description="Masked Style Transfer with Voronoi Regions")
    parser.add_argument('--num_points', type=int, help='Number of points for Voronoi diagram', required=True)
    parser.add_argument('--output_dirs', nargs='+', help='output directories for results', required=True)
    args = parser.parse_args()

    # Define arguments here instead of using argparse
    content_dirs = ['/home/bhamscher/datasets/Cityscapes/leftImg8bit/train', 
                    '/home/bhamscher/datasets/Cityscapes/leftImg8bit/val']
    style_dir = '/home/bhamscher/results/patches_greater_100'
    output_dirs = args.output_dirs
    mask_dirs = ['/home/bhamscher/datasets/Cityscapes/gtFine/train', 
                 '/home/bhamscher/datasets/Cityscapes/gtFine/val']
    alpha = 1.0
    content_size = 0
    style_size = 256
    crop = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stylize_proportion = 1.0

    for content_dir, output_dir, mask_dir in zip(content_dirs, output_dirs, mask_dirs):
        print(f"Stylizing Images in {content_dir}.")
        content_dir = Path(content_dir).resolve()
        style_dir = Path(style_dir).resolve()
        output_dir = Path(output_dir).resolve()
        mask_dir = Path(mask_dir).resolve()

        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
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
        styles = collect_style_images(style_dir)
        print(f'Found styles for {len(styles)} classes in {style_dir}')

        # Disable decompression bomb errors
        Image.MAX_IMAGE_PIXELS = None
        skipped_imgs = []

        # Define transforms
        content_tf = input_transform(content_size, crop)
        style_tf = input_transform(style_size, 0)

        # Process images
        num_images = len(image_pairs)
        # num_images = 2  # For testing
        with tqdm(total=num_images) as pbar:
            for (content_path, (panoptic_mask_path, semantic_mask_path)) in list(image_pairs.items())[:num_images]:
                try:
                    # Load content image and masks
                    content_img = Image.open(content_path).convert('RGB')
                    content_tensor = content_tf(content_img)
                    
                    # Get correct dimensions from the content image
                    h, w = content_img.size[1], content_img.size[0]  # PIL Image size is (width, height)
                    
                    # Generate Voronoi mask with correct dimensions
                    voronoi_mask = generate_voronoi_mask(h, w, num_points=args.num_points, available_IDs=list(range(19)))
                    
                    # Convert mask to tensor
                    voronoi_mask = torch.from_numpy(voronoi_mask).float()
                    
                    # Initialize result tensor
                    result = torch.zeros_like(content_tensor).to(device)
                    new_style_mask = np.zeros_like(voronoi_mask.numpy())
                    
                    # Process each unique ID
                    unique_ids = torch.unique(voronoi_mask)
                    for id in unique_ids:
                        id_int = int(id.item())  # Convert tensor to integer
                        if str(id_int) not in styles:
                            continue
                            
                        # Select random style
                        style_path = random.choice(styles[str(id_int)])
                        
                        # Load and preprocess style image
                        style_img = Image.open(style_path).convert('RGB')
                        style_tensor = style_tf(style_img)
                        
                        # Create mask for current region
                        id_mask = (voronoi_mask == id).float()
                        mask_3d = id_mask.unsqueeze(0).repeat(3, 1, 1)
                        
                        # Perform style transfer
                        stylized_output = stylize_panoptic(content_tensor, style_tensor, vgg, decoder, alpha, device)
                        
                        # Apply mask and add to result
                        mask_3d = mask_3d.to(device)
                        result = result + (mask_3d * stylized_output.squeeze(0))
                        
                        # Update style mask
                        new_style_mask[voronoi_mask.numpy() == id_int] = id_int

                    # Save results
                    content_path = Path(content_path)
                    rel_path = content_path.relative_to(content_dir)
                    out_dir = output_dir.joinpath(rel_path.parent)
                    out_dir.mkdir(parents=True, exist_ok=True)

                    # Save stylized image
                    content_name = content_path.stem
                    out_filename = f"{content_name}{content_path.suffix}"
                    output_name = out_dir.joinpath(out_filename)
                    save_image(result.unsqueeze(0), output_name, padding=0)

                    # Save style masks
                    new_mask_name = f"{content_name}_style_mask.png"
                    new_mask_path = out_dir.joinpath(new_mask_name)
                    Image.fromarray(new_style_mask.astype(np.uint8)).save(new_mask_path)

                    # Create and save RGB style mask
                    rgb_array = np.zeros((h, w, 3), dtype=np.uint8)
                    for train_id in trainId2label.keys():
                        if train_id in trainId2label:  # Check if ID exists in mapping
                            rgb_color = trainId2label[train_id].color
                            mask = (new_style_mask == train_id)
                            rgb_array[mask] = rgb_color

                    rgb_image = Image.fromarray(rgb_array)
                    rgb_image_name = f"{content_name}_style_mask_rgb.png"
                    rgb_image_path = out_dir.joinpath(rgb_image_name)
                    rgb_image.save(rgb_image_path)

                    print(f"Processed {content_path}")
                    print(f"Saved stylized image to {output_name}")
                    print(f"Saved style mask to {new_mask_path}")
                    print(f"Saved RGB style mask to {rgb_image_path}")

                except Exception as e:
                    print(f'Skipping stylization of {content_path} due to an error: {str(e)}')
                    skipped_imgs.append(content_path)
                    continue
                finally:
                    pbar.update(1)

if __name__ == '__main__':
    main()