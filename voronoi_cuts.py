import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from PIL import Image, ImageDraw

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

def generate_voronoi_mask(height, width, num_points):
    points = np.random.rand(num_points, 2) * [width, height]
    vor = Voronoi(points)
    regions, vertices = voronoi_finite_polygons_2d(vor)
    mask = np.zeros((height, width), dtype=np.int32)

    for i, region in enumerate(regions):
        polygon = vertices[region]
        polygon = [(int(x), int(y)) for x, y in polygon]
        img = Image.new('L', (width, height), 0)
        ImageDraw.Draw(img).polygon(polygon, outline=1, fill=i + 1)
        mask += np.array(img)

    return mask

def visualize_voronoi_mask(mask, output_path):
    unique_ids = np.unique(mask)
    height, width = mask.shape
    rgb_array = np.zeros((height, width, 3), dtype=np.uint8)

    np.random.seed(0)
    colors = {id: np.random.randint(0, 255, 3) for id in unique_ids if id != 0}

    for id in unique_ids:
        if id == 0:
            continue
        rgb_array[mask == id] = colors[id]

    img = Image.fromarray(rgb_array)
    img.save(output_path)

def main():
    height = 1024
    width = 2048
    num_cells_list = [10, 20, 50, 100, 200]

    for i, num_cells in enumerate(num_cells_list):
        voronoi_mask = generate_voronoi_mask(height, width, num_cells)
        output_path = f"voronoi_diagram_{num_cells}_cells.png"
        visualize_voronoi_mask(voronoi_mask, output_path)
        print(f"Saved Voronoi diagram with {num_cells} cells to {output_path}")

if __name__ == "__main__":
    main()