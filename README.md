# README — Tile Stitching & Alignment Toolkit

This repository provides a modular pipeline to stitch and geometrically correct a mosaic of microscope tiles using stage coordinates, neighbor detection, SIFT-based matching on overlap crops, pixel→micron conversion, and a weighted least-squares optimization to refine tile positions.

---

## 1) High-level flow (what runs when)

1. **Load tile positions** from a CSV into a clean DataFrame with standardized columns `tile_name, x, y`. 
2. **Infer neighbors (8-connected)** from stage coordinates with DBSCAN gating and KD-Tree range queries to avoid cross-cluster false links; estimate nominal step sizes. 
3. **Load tile images** (grayscale or color) keyed by `tile_name` and paired with their stage coordinates for downstream cropping. 
4. **Crop overlap regions** per direction (left/right/up/down + 4 diagonals) to focus matching on informative borders. 
5. **Match overlap crops with SIFT**, compute per-pair affine transforms and inlier counts via RANSAC, and collect displacements and quality stats. 
6. **Convert pixel displacements to microns**, estimating a global microns-per-pixel from stage deltas and affine translations; filter out invalid results. 
7. **Solve a weighted position refinement**, building a linear system from pairwise rules (dx, dy) + Tikhonov regularization + an optional anchor tile; multiple weighting schemes are supported. 
8. **Package imports** are exposed in the package’s `__init__.py` to simplify user code. 

---

## 2) Data model (shared assumptions)

* **Tiles** are identified by `tile_name` and have **stage coordinates** `(x, y)` in microns (or stage units). The CSV loader normalizes to integer `x, y` columns and trims names. 
* **Neighbor graph** is 8-connected (cardinals + diagonals) when tiles fall within tolerances of the median grid steps (`step_x, step_y`). Outliers/noise can be clustered away by DBSCAN. 
* **Images** are read from a directory and stored as a dict: `tile_name -> ((x, y), image)`, enabling crop logic to use both image content and known position. 
* **Overlap crops** are derived from per-direction slices controlled by `overlap_fraction` and `extra_factor`. 
* **Matches** store SIFT keypoints, “good” matches (Lowe’s ratio), estimated affine, displacement `(dx, dy)`, rotation, and inlier counts. 
* **Unit conversion** computes a single global microns-per-pixel using stage deltas between a matched pair and the affine translation magnitude, then applies it to all `(dx, dy)`. 
* **Optimization** treats each pairwise relation as a “rule” with weight `w`, and solves a regularized least-squares system for refined `(x, y)` of all tiles, optionally anchoring one tile. 

---

## 3) Module-by-module details

### A) `LoadCoordinates.py` — CSV → DataFrame

* **Purpose:** Read microscope tile stage coordinates and standardize field names.
* **Key behavior:** Maps CSV columns `Tile, X, Y` → `tile_name, x, y`; trims names; casts coordinates to `int`.
* **Output:** `pd.DataFrame[['tile_name','x','y']]`. 

### B) `NeighborDetection.py` — Robust 8-neighbor inference

* **Purpose:** Build an 8-connected neighbor map using geometry rather than filenames.
* **How it works:**

  * Estimates median spacing along each axis (`step_x`, `step_y`).
  * Uses DBSCAN to isolate clusters and prevent linking distant tiles.
  * Within each cluster, KD-Tree radius queries find nearby tiles; directional assignment uses sign and tolerance checks on `(dx, dy)` against the step sizes.
  * Optionally draws a vector visualization of neighbor directions.
* **Returns:** `(neighbor_map, step_x, step_y)`. 

### C) `LoadImages.py` — Read tiles from disk

* **Purpose:** Load grayscale or color images for a set of tiles listed in the DataFrame.
* **I/O shape:** Returns a dict `tile_name -> ((x, y), image)` for either grayscale (`IMREAD_GRAYSCALE`) or color (`IMREAD_COLOR`), emitting warnings for failures.
* **Notes:** Relies on exact `tile_name` → filename matches; trims `tile_name` to avoid whitespace mismatches. 

### D) `CropOverlapRegions.py` — Directional crops

* **Purpose:** Extract only the likely overlapping borders to make SIFT faster and cleaner.
* **Mechanics:**

  * Cleans both `tile_data` and `neighbors` keys/values (trimming names).
  * Computes crop widths/heights as `overlap_fraction + extra_factor` of image dimensions.
  * Produces per-tile dict of patches keyed by direction.
* **Directions covered:** left, right, up, down, top\_left, top\_right, bottom\_left, bottom\_right. 

### E) `MatchTilePairs.py` — SIFT matching on overlap patches

* **Purpose:** For each tile/direction, match its cropped patch to the opposite-side crop of its neighbor and estimate an affine transform.
* **Pipeline specifics:**

  * SIFT feature detection + descriptors; BFMatcher with KNN and Lowe’s ratio test (`ratio_thresh`).
  * Opposite-direction mapping ensures correct patch pairing (e.g., tile A’s “right” vs tile B’s “left”).
  * Computes `AffinePartial2D` with RANSAC; records translation `(dx, dy)`, rotation in degrees, and `num_inliers`.
  * Skips pairs with too few matches (`min_matches`).
* **Outputs:** A dict keyed by `(tile_i, tile_j)` with per-pair metadata and quality stats. 

### F) `SIFTConversion.py` — Pixel → Microns conversion & filtering

* **Purpose:** Convert per-pair pixel displacements to microns using an estimated global scale; drop invalid pairs.
* **Method:**

  * Finds the first pair with a valid affine matrix and known stage positions; computes microns-per-pixel from the ratio of stage delta magnitude to affine translation magnitude.
  * If no valid pair exists, defaults to `1.0` and warns.
  * Applies conversion to all pairs and returns a filtered dict with `dx_microns`, `dy_microns`, and the chosen scale. 

### G) `Optimization.py` — Weighted least-squares position refinement

* **Purpose:** Integrate all pairwise displacements into a single, globally consistent correction of tile coordinates.
* **Steps:**

  1. **Weights:** Multiple schemes (`raw_inliers`, `sqrt_inliers`, `inlier_ratio`, `hybrid`, `log_capped`, `uniform`) computed from match quality; optional normalization.
  2. **Rules:** For each matched pair `(i, j)`, build constraints enforcing `x_j - x_i ≈ dx_microns` and `y_j - y_i ≈ dy_microns`, scaled by weight.
  3. **Regularization:** Penalize deviation from noisy stage points with λ (Tikhonov), stabilizing under sparse/uneven coverage.
  4. **Anchoring:** Optionally fix the first point to remove global translation ambiguity.
  5. **Solve:** Linear least-squares; optionally visualize original vs corrected positions and residual vectors.
* **Returns:** `corrected_points` and residuals. A convenience `run(...)` composes the full workflow. 

### H) `__init__.py` — Public API surface

* **Purpose:** Re-exports core entry points for convenient imports from the package root (coordinates loading, neighbor detection, image loading, cropping, conversion). 

---

## 4) Practical guidance & gotchas

* **File naming:** `tile_name` must match the on-disk image filename; the loaders do a `.strip()` on names, but not more. If images fail to load, verify exact names and extensions. 
* **Grid irregularities:** If stage coordinates are noisy or sparse, DBSCAN parameters (`eps`, `min_samples`) and neighbor tolerance may need tuning to avoid cross-row/column links. 
* **Crop sizing:** Increase `overlap_fraction` or `extra_factor` if matches are weak or borders are too thin; reduce them for speed once stability is confirmed. 
* **SIFT thresholds:** If you get many “No descriptors” or too few matches, consider adjusting `ratio_thresh` or ensuring sufficient texture in the overlap zones. 
* **Scale estimation:** The microns-per-pixel is derived from the first valid pair; ensure at least one reliable match exists. If not, the module falls back to 1.0 (your optimization will still run, but in pixel units). 
* **Weighting choice:**

  * Use `hybrid` (default) to emphasize both inlier count and match purity.
  * Switch to `uniform` for ablation or to debug the effect of matcher noise.
  * `log_capped` can temper a few very strong links dominating the solution. 
* **Regularization λ:** Larger λ keeps refined points closer to stage coordinates (useful when matches are sparse or uneven); smaller λ lets matches drive bigger corrections. 
* **Anchoring:** Keep `fix_tile=True` unless you explicitly post-shift the whole solution; otherwise the system is underdetermined for translation. 

---

## 5) Typical end-to-end usage (conceptual)

1. Load CSV → `df` with `tile_name, x, y`. 
2. Infer `(neighbor_map, step_x, step_y)` from `df`. 
3. Load grayscale images → `tile_data`. 
4. Crop overlaps → `cropped_patches`. 
5. Match neighbors → `match_results` with affine and inlier stats. 
6. Convert to microns + filter → `filtered_matches` and `microns_per_pixel`. 
7. Optimize → refined coordinates `corrected_points` (optionally plot). 
8. (Optional) Import helpers directly from package root per `__init__.py`. 

---

## 6) Configuration cheat-sheet

* **Neighbor detection:** `tolerance` (fraction of step), `dbscan_eps`, `dbscan_min_samples`, `visualize`. 
* **Crops:** `overlap_fraction` (e.g., 0.15), `extra_factor` (e.g., 0.10). 
* **SIFT matching:** `ratio_thresh` (e.g., 0.75), `min_matches` (e.g., 3). 
* **Conversion:** `anchor` (reserved), returns `microns_per_pixel`. 
* **Optimization:** `weight_method`, `normalize`, `lambda_reg`, `fix_tile`, `visualize`. 

---

## 7) Outputs you should expect

* **Neighbor graph:** dict of `tile_name -> directions -> neighbor_name`. 
* **Overlap crops:** per-tile dict of border images by direction. 
* **Match summary:** per-pair metadata including `num_matches`, `num_inliers`, `affine_matrix`, `(dx, dy)`, `rotation_deg`. 
* **Converted matches:** `dx_microns`, `dy_microns`, global `microns_per_pixel`. 
* **Refined coordinates:** `corrected_points` aligned in a consistent global frame, plus residuals. 

---

## 8) Troubleshooting

* **“Missing patch” or `None` neighbor:** Check neighbor detection tolerances and clustering; ensure tiles actually overlap at expected step sizes.  
* **Few or zero SIFT matches:** Increase crop size, verify image contrast on borders, relax `ratio_thresh`, or switch to grayscale inputs.   
* **Scale seems off:** Ensure at least one high-quality pair exists for pixel→micron estimation; otherwise you’ll see a warning and a fallback to 1.0. 
* **Warped global solution:** Try `log_capped` or `uniform` weights, increase `lambda_reg`, or check for mislabeled neighbors that inject contradictory rules.  

---
