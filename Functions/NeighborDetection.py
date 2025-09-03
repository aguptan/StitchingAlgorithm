from typing import Tuple, Dict
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KDTree
import pandas as pd

def find_robust_tile_neighbors_with_diagonals(
    df: pd.DataFrame,
    tolerance: float = 0.25,
    dbscan_eps: float = 2000,
    dbscan_min_samples: int = 2,
    visualize: bool = False
) -> Tuple[Dict[str, Dict[str, str]], float, float]:
    """
    Find 8-connected neighbors for microscope tiles using stage coordinates.

    Args:
        df (DataFrame): Tile positions with columns ['tile_name', 'x', 'y'].
        tolerance (float): Fractional tolerance on spacing.
        dbscan_eps (float): DBSCAN radius.
        dbscan_min_samples (int): Minimum samples per cluster.
        visualize (bool): Whether to visualize the result.

    Returns:
        neighbor_map: Dict mapping each tile to its 8 neighbors.
        step_x: Estimated median X step.
        step_y: Estimated median Y step.
    """
    coords = df[['x', 'y']].values
    names = df['tile_name'].values

    # Estimate step size
    unique_x = np.sort(df['x'].unique())
    unique_y = np.sort(df['y'].unique())
    step_x = float(np.median(np.diff(unique_x)))
    step_y = float(np.median(np.diff(unique_y)))
    # print(f"Estimated step_x = {step_x}, step_y = {step_y}")

    # Cluster tiles to prevent false neighbors across sparse regions
    clustering = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
    df['cluster'] = clustering.fit_predict(coords)

    neighbor_map = {}

    for cluster_id in sorted(df['cluster'].unique()):
        if cluster_id == -1:
            continue

        cluster_df = df[df['cluster'] == cluster_id].reset_index(drop=True)
        cluster_coords = cluster_df[['x', 'y']].values
        cluster_names = cluster_df['tile_name'].values

        tree = KDTree(cluster_coords)
        search_radius = np.sqrt(step_x**2 + step_y**2) * (1 + tolerance)
        indices = tree.query_radius(cluster_coords, r=search_radius)

        for idx, neighbors in enumerate(indices):
            name = cluster_names[idx]
            x0, y0 = cluster_coords[idx]
            neighbor_map[name] = {
                'up': None, 'down': None, 'left': None, 'right': None,
                'top_left': None, 'top_right': None,
                'bottom_left': None, 'bottom_right': None
            }

            for n_idx in neighbors:
                if n_idx == idx:
                    continue
                x1, y1 = cluster_coords[n_idx]
                dx = x1 - x0
                dy = y1 - y0
                other = cluster_names[n_idx]

                # Cardinal
                if dx > 0 and abs(dx - step_x) <= step_x * tolerance and abs(dy) <= step_y * tolerance:
                    neighbor_map[name]['right'] = other
                elif dx < 0 and abs(dx + step_x) <= step_x * tolerance and abs(dy) <= step_y * tolerance:
                    neighbor_map[name]['left'] = other
                elif dy > 0 and abs(dy - step_y) <= step_y * tolerance and abs(dx) <= step_x * tolerance:
                    neighbor_map[name]['down'] = other
                elif dy < 0 and abs(dy + step_y) <= step_y * tolerance and abs(dx) <= step_x * tolerance:
                    neighbor_map[name]['up'] = other

                # Diagonals
                elif dx > 0 and dy > 0 and abs(dx - step_x) <= step_x * tolerance and abs(dy - step_y) <= step_y * tolerance:
                    neighbor_map[name]['bottom_right'] = other
                elif dx < 0 and dy > 0 and abs(dx + step_x) <= step_x * tolerance and abs(dy - step_y) <= step_y * tolerance:
                    neighbor_map[name]['bottom_left'] = other
                elif dx > 0 and dy < 0 and abs(dx - step_x) <= step_x * tolerance and abs(dy + step_y) <= step_y * tolerance:
                    neighbor_map[name]['top_right'] = other
                elif dx < 0 and dy < 0 and abs(dx + step_x) <= step_x * tolerance and abs(dy + step_y) <= step_y * tolerance:
                    neighbor_map[name]['top_left'] = other

    # Optional visualization
    if visualize:
        plt.figure(figsize=(10, 10))
        plt.scatter(df['x'], df['y'], c='blue', label='Tiles', zorder=3)

        offset = 30
        arrow_length_frac = 0.3
        arrow_colors = {
            'up': 'green', 'down': 'green',
            'left': 'red', 'right': 'red',
            'top_left': 'purple', 'top_right': 'purple',
            'bottom_left': 'orange', 'bottom_right': 'orange'
        }

        for name, neighbors in neighbor_map.items():
            x0, y0 = df[df['tile_name'] == name][['x', 'y']].values[0]

            for direction, neighbor in neighbors.items():
                if neighbor:
                    x1, y1 = df[df['tile_name'] == neighbor][['x', 'y']].values[0]
                    dx = x1 - x0
                    dy = y1 - y0
                    length = np.hypot(dx, dy)
                    if length == 0:
                        continue

                    ux, uy = dx / length, dy / length
                    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
                    perp_x, perp_y = -uy, ux
                    ox, oy = perp_x * offset, perp_y * offset
                    mx += ox
                    my += oy

                    arrow_dx = ux * length * arrow_length_frac
                    arrow_dy = uy * length * arrow_length_frac

                    plt.arrow(
                        mx - arrow_dx / 2, my - arrow_dy / 2,
                        arrow_dx, arrow_dy,
                        head_width=step_x * 0.05,
                        head_length=step_y * 0.05,
                        length_includes_head=True,
                        fc=arrow_colors.get(direction, 'gray'),
                        ec=arrow_colors.get(direction, 'gray'),
                        alpha=0.8,
                        zorder=2
                    )

        plt.gca().invert_yaxis()
        plt.title("Centered Directional Arrows Between Neighbor Tiles")
        plt.axis("equal")
        plt.legend()
        plt.show()

    return neighbor_map, step_x, step_y
