import os
import cv2
import numpy as np
from typing import Dict, Tuple
import pandas as pd

def load_grayscale_images_from_df(df: pd.DataFrame, image_dir: str) -> Dict[str, Tuple[Tuple[int, int], np.ndarray]]:
    """
    Load grayscale images for all tiles listed in a DataFrame.

    Args:
        df (pd.DataFrame): Must contain columns ['tile_name', 'x', 'y']
        image_dir (str): Directory containing tile images

    Returns:
        Dict[str, Tuple[(x, y), grayscale_image]]
    """
    tile_data = {}
    for _, row in df.iterrows():
        tile_name = row['tile_name'].strip()
        x, y = int(row['x']), int(row['y'])
        img_path = os.path.join(image_dir, tile_name)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            tile_data[tile_name] = ((x, y), img)
        else:
            print(f"[Grayscale] Warning: failed to load {tile_name}")

    return tile_data

def load_color_images_from_df(df: pd.DataFrame, image_dir: str) -> Dict[str, Tuple[Tuple[int, int], np.ndarray]]:
    """
    Load color images for all tiles listed in a DataFrame.

    Args:
        df (pd.DataFrame): Must contain columns ['tile_name', 'x', 'y']
        image_dir (str): Directory containing tile images

    Returns:
        Dict[str, Tuple[(x, y), color_image]]
    """
    tile_data = {}
    for _, row in df.iterrows():
        tile_name = row['tile_name'].strip()
        x, y = int(row['x']), int(row['y'])
        img_path = os.path.join(image_dir, tile_name)

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is not None:
            tile_data[tile_name] = ((x, y), img)
        else:
            print(f"[Color] Warning: failed to load {tile_name}")

    return tile_data
