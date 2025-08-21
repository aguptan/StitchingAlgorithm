import numpy as np

def convert_and_filter_sift_displacements_to_microns(match_results, tile_df, anchor='top-left'):   
    """
    Compute microns-per-pixel and convert SIFT dx/dy to microns. Drop invalid entries (dx or dy is N/A or None).

    Args:
        match_results (dict): Output of match_cropped_tile_pairs
        tile_df (pd.DataFrame): Contains 'tile_name', 'x', 'y'
        anchor (str): Anchor point for transformation (currently unused but kept for future flexibility)

    Returns:
        filtered_results: dict of valid matches with added 'dx_microns', 'dy_microns'
        dropped: list of dropped match keys
    """
    stage_coords = {row['tile_name']: (row['x'], row['y']) for _, row in tile_df.iterrows()}
    microns_per_pixel = None

    # Step 1: Estimate microns-per-pixel once
    for (tile1, tile2), result in match_results.items():
        M = result.get("affine_matrix", None)
        if M is not None and tile1 in stage_coords and tile2 in stage_coords:
            ref_coords = np.array(stage_coords[tile1], dtype=np.float32)
            mov_coords = np.array(stage_coords[tile2], dtype=np.float32)
            stage_delta = ref_coords - mov_coords

            pixel_translation_vector = np.array([M[0, 2], M[1, 2]], dtype=np.float32)
            pixel_magnitude = np.linalg.norm(pixel_translation_vector)
            micron_magnitude = np.linalg.norm(stage_delta)

            if pixel_magnitude > 0:
                microns_per_pixel = micron_magnitude / pixel_magnitude
                break

    if microns_per_pixel is None:
        print("[Warning] Could not estimate microns-per-pixel. Using default = 1.0")
        microns_per_pixel = 1.0

    # Step 2: Apply conversion and filter
    filtered_results = {}
    dropped = []

    for key, result in match_results.items():
        dx, dy = result.get("dx", None), result.get("dy", None)

        if dx is None or dy is None:
            dropped.append(key)
            continue
        if isinstance(dx, str) and ("N/A" in dx or "na" in dx.lower()):
            dropped.append(key)
            continue
        if isinstance(dy, str) and ("N/A" in dy or "na" in dy.lower()):
            dropped.append(key)
            continue

        result["dx_microns"] = float(dx) * microns_per_pixel
        result["dy_microns"] = float(dy) * microns_per_pixel
        result["microns_per_pixel"] = microns_per_pixel
        filtered_results[key] = result

    # Step 3: Print dropped entries
    # if dropped:
        # print(f"\n[Filtered] Dropped {len(dropped)} invalid entries due to missing dx/dy:")
        # for key in dropped:
            # print(f"  - {key[0]} → {key[1]}")
    # else:
        # print("[Filtered] No entries dropped.")

    return microns_per_pixel, filtered_results