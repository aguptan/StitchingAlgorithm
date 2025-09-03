from typing import Dict, Tuple
import numpy as np

def crop_tile_overlap_regions(
    tile_data: Dict[str, Tuple[Tuple[int, int], np.ndarray]],
    neighbors: Dict[str, Dict[str, str]],
    overlap_fraction: float = 0.15,
    extra_factor: float = 0.10
) -> Dict[str, Dict[str, np.ndarray]]:
    
    
    cropped = {}
    clean_tile_data = {k.strip(): v for k, v in tile_data.items()}
    clean_neighbors = {
        k.strip(): {d: (v.strip() if v else None) for d, v in nd.items()}
        for k, nd in neighbors.items()
    }

    for tile_name, ((_, _), img) in clean_tile_data.items():
        h, w = img.shape[:2]
        ox = int(w * (overlap_fraction + extra_factor))
        oy = int(h * (overlap_fraction + extra_factor))

        cropped[tile_name] = {}
        for direction, neighbor in clean_neighbors.get(tile_name, {}).items():
            if neighbor is None:
                continue

            if direction == 'right':
                patch = img[:, w - ox:]
            elif direction == 'left':
                patch = img[:, :ox]
            elif direction == 'up':
                patch = img[:oy, :]
            elif direction == 'down':
                patch = img[h - oy:, :]
            elif direction == 'top_left':
                patch = img[:oy, :ox]
            elif direction == 'top_right':
                patch = img[:oy, w - ox:]
            elif direction == 'bottom_left':
                patch = img[h - oy:, :ox]
            elif direction == 'bottom_right':
                patch = img[h - oy:, w - ox:]
            else:
                continue

            cropped[tile_name][direction] = patch

    return cropped
