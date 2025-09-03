import sys
import os
from dataclasses import dataclass
from typing import Dict, List
import pandas as pd
DATASETS_ROOT = r"C:\Users\agupt\Desktop\Shoykhet_Lab\Stitching\Optimization\OptimizationDataSet"
FUNCTIONS_DIR = r"C:\Users\agupt\Desktop\Shoykhet_Lab\Stitching\Algorithm\Functions"

if FUNCTIONS_DIR not in sys.path:
    sys.path.insert(0, FUNCTIONS_DIR)
    
from LoadCoordinates import load_tile_coordinates
from NeighborDetection import find_robust_tile_neighbors_with_diagonals
from LoadImages import load_grayscale_images_from_df
from CropOverlapRegions import crop_tile_overlap_regions
from MatchTilePairs import match_cropped_tile_pairs
from SIFTConversion import convert_and_filter_sift_displacements_to_microns
from Optimization import Optimization




# ----------------------------------
# 1. Data structures & Data loading
# ----------------------------------
@dataclass
class DatasetEntry:
    dataset_id: str
    csv_path: str
    image_dir: str
    stain: str
    white_balanced: bool
    domain: str  # e.g., "H&E+WB", "Nissl+noWB", "Unknown+WB"

@dataclass
class PreloadedDataset:
    dataset_id: str
    stain: str
    white_balanced: bool
    domain: str
    image_dir: str
    tile_df: pd.DataFrame
    neighbors: dict
    tiles_gray: dict
    overlaps: dict

def build_fixed_registry(root: str) -> List[DatasetEntry]:
    spec = [
        ("HnE_Stain",        "H&E",    True),
        ("Nissl_Stain_nWB1", "Nissl",  False),
        ("Nissl_Stain_WB",   "Nissl",  True),
        ("Unknown_Stain",    "Unknown", True),
    ]
    entries: List[DatasetEntry] = []
    for folder_name, stain, wb in spec:
        folder = os.path.join(root, folder_name)
        csv_path = os.path.join(folder, "TileConfiguration.csv")
        image_dir = folder
        if os.path.isdir(folder) and os.path.isfile(csv_path):
            domain = f"{stain}+{'WB' if wb else 'noWB'}"
            entries.append(DatasetEntry(
                dataset_id=folder_name,
                csv_path=csv_path,
                image_dir=image_dir,
                stain=stain,
                white_balanced=wb,
                domain=domain
            ))
        else:
            print(f"[Registry][SKIP] {folder_name} (missing folder or TileConfiguration.csv)")
    if not entries:
        raise RuntimeError("No datasets found. Check folders and TileConfiguration.csv files.")
    return entries


def preload_dataset(
    entry: DatasetEntry,
    *,
    tolerance: float = 0.25,
    dbscan_eps: float = 2000.0,
    dbscan_min_samples: int = 2,
    overlap_fraction: float = 0.25,
    extra_factor: float = 0.10,
    visualize_neighbors: bool = False
) -> PreloadedDataset:
    print(f"[Preload] {entry.dataset_id}  stain={entry.stain}  WB={'Yes' if entry.white_balanced else 'No'}  domain={entry.domain}")

    # 1) Coordinates
    tile_df = load_tile_coordinates(entry.csv_path)

    # 2) Neighbor graph
    neighbors, _, _ = find_robust_tile_neighbors_with_diagonals(
        tile_df,
        tolerance=tolerance,
        dbscan_eps=dbscan_eps,
        dbscan_min_samples=dbscan_min_samples,
        visualize=visualize_neighbors
    )

    # 3) Grayscale tiles
    tiles_gray = load_grayscale_images_from_df(tile_df, entry.image_dir)

    # 4) Cropped overlaps
    overlaps = crop_tile_overlap_regions(
        tile_data=tiles_gray,
        neighbors=neighbors,
        overlap_fraction=overlap_fraction,
        extra_factor=extra_factor
    )

    return PreloadedDataset(
        dataset_id=entry.dataset_id,
        stain=entry.stain,
        white_balanced=entry.white_balanced,
        domain=entry.domain,
        image_dir=entry.image_dir,
        tile_df=tile_df,
        neighbors=neighbors,
        tiles_gray=tiles_gray,
        overlaps=overlaps
    )


def preload_all_for_cv(
    registry: List[DatasetEntry],
    *,
    tolerance: float = 0.25,
    dbscan_eps: float = 2000.0,
    dbscan_min_samples: int = 2,
    overlap_fraction: float = 0.25,
    extra_factor: float = 0.10
) -> Dict[str, PreloadedDataset]:
    out: Dict[str, PreloadedDataset] = {}
    for entry in registry:
        try:
            ds = preload_dataset(
                entry,
                tolerance=tolerance,
                dbscan_eps=dbscan_eps,
                dbscan_min_samples=dbscan_min_samples,
                overlap_fraction=overlap_fraction,
                extra_factor=extra_factor,
                visualize_neighbors=False
            )
            out[entry.dataset_id] = ds
        except Exception as e:
            print(f"[Preload][SKIP] {entry.dataset_id}: {type(e).__name__}: {e}")
    if not out:
        raise RuntimeError("No datasets successfully preloaded.")
    print(f"[Preload] Completed: {len(out)} dataset(s).")
    return out


def run_stitching_pipeline_dataset_aware(
    dataset_id: str,
    preloaded_by_id: dict,
    match_cache: dict,
    lambda_reg: float,
    weight_method: str,
    ratio_thresh: float,
    min_matches: int
):
    ds = preloaded_by_id[dataset_id]
    tile_df = ds.tile_df
    neighbors = ds.neighbors
    cropped_patches = ds.overlaps

    if cropped_patches is None:
        raise ValueError(
            f"Dataset '{dataset_id}' has no preloaded overlaps. "
            f"Preload with overlaps enabled, or add an on-demand crop path."
        )

    match_key = (dataset_id, ratio_thresh, min_matches)

    if match_key in match_cache:
        match_results, filtered_results = match_cache[match_key]
    else:
        match_results = match_cropped_tile_pairs(
            cropped_patches=cropped_patches,
            neighbors=neighbors,
            ratio_thresh=ratio_thresh,
            min_matches=min_matches
        )
        _, filtered_results = convert_and_filter_sift_displacements_to_microns(
            match_results,
            tile_df
        )
        match_cache[match_key] = (match_results, filtered_results)

    opt = Optimization(tile_df, filtered_results, neighbors)
    corrected_points, _ = opt.run(
        lambda_reg=lambda_reg,
        weight_method=weight_method,
        normalize=(weight_method not in ["none", "uniform"]),
        fix_tile=True,
        visualize=False
    )
    return corrected_points, opt.noisy_points, opt.A, opt.b, filtered_results

if __name__ == "__main__":
    # 1) Build registry and preload once (your existing calls)
    registry = build_fixed_registry(DATASETS_ROOT)
    print(f"[Registry] {len(registry)} dataset(s) registered.")

    preloaded_by_id = preload_all_for_cv(
        registry,
        tolerance=0.25,
        dbscan_eps=2000.0,
        dbscan_min_samples=2,
        overlap_fraction=0.25,
        extra_factor=0.10
    )

    print("\n[Summary] Ready for CV:")
    for dsid, ds in preloaded_by_id.items():
        n_tiles = len(ds.tile_df)
        n_edges = len(ds.neighbors) if hasattr(ds.neighbors, "__len__") else "unknown"
        print(f"  - {dsid:>18} | stain={ds.stain:<7} | WB={'Yes' if ds.white_balanced else 'No':<3} | domain={ds.domain:<12} | tiles={n_tiles:<5} | nodes={n_edges}")

    # 2) Basic per-dataset run (unchanged sanity check)
    match_cache = {}
    lambda_reg   = 1.0
    weight_method = "uniform"
    ratio_thresh = 0.75
    min_matches  = 8

    print("\n[Run] Evaluating datasets with the same hyperparameters:")
    for dataset_id in preloaded_by_id.keys():
        print(f"  -> {dataset_id}")
        corrected_points, noisy_points, A, b, filtered_results = run_stitching_pipeline_dataset_aware(
            dataset_id=dataset_id,
            preloaded_by_id=preloaded_by_id,
            match_cache=match_cache,
            lambda_reg=lambda_reg,
            weight_method=weight_method,
            ratio_thresh=ratio_thresh,
            min_matches=min_matches
        )
        n_links = len(filtered_results) if hasattr(filtered_results, "__len__") else "unknown"
        print(f"     links_used={n_links}, corrected_points={getattr(corrected_points, 'shape', None)}, noisy_points={getattr(noisy_points, 'shape', None)}")

    # Cache sanity check
    first_id = next(iter(preloaded_by_id.keys()))
    match_key = (first_id, ratio_thresh, min_matches)
    print(f"\n[Cache] Key exists for first dataset? {match_key in match_cache}")

 