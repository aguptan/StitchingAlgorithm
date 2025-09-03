import sys
from typing import Dict, List, Tuple, Any
import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

DATASETS_ROOT = r"C:\Users\agupt\Desktop\Shoykhet_Lab\Stitching\Optimization\OptimizationDataSet"
FUNCTIONS_DIR = r"C:\Users\agupt\Desktop\Shoykhet_Lab\Stitching\Algorithm\Functions"

if FUNCTIONS_DIR not in sys.path:
    sys.path.insert(0, FUNCTIONS_DIR)

from DatasetLoading import (
    build_fixed_registry,
    preload_all_for_cv,
    run_stitching_pipeline_dataset_aware,
)

from CrossValidationTesting import (
    build_outer_folds,
    build_inner_folds
)


import random

# -----------------------------
# Global Seed Control
# -----------------------------
GLOBAL_SEED = 123
np.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)

try:
    import torch
    torch.manual_seed(GLOBAL_SEED)
    torch.cuda.manual_seed_all(GLOBAL_SEED)
except ImportError:
    pass

# ----------------------------------
# 3. Objective Creation
# ----------------------------------
def compute_score(corrected_points, noisy_points, A, b, alpha=0.1):
    Ax = A @ corrected_points.flatten()
    constraint_error = np.sum((Ax - b) ** 2) / A.shape[0]
    displacement_error = np.sum((corrected_points - noisy_points) ** 2) / A.shape[0]
    return constraint_error + alpha * displacement_error, constraint_error, displacement_error

def compute_link_dropout_penalty_ratio(neighbor_map, filtered_matches, cardinal_ratio=5.0, base_weight=1.0):
    diagonal_dirs = {'top_left', 'top_right', 'bottom_left', 'bottom_right'}
    penalty = 0.0
    dropped_links = []

    for tile, neighbors in neighbor_map.items():
        for direction, neighbor in neighbors.items():
            if neighbor is None:
                continue

            key1 = (tile, neighbor)
            key2 = (neighbor, tile)

            if key1 not in filtered_matches and key2 not in filtered_matches:
                dropped_links.append((tile, direction, neighbor))
                if direction in diagonal_dirs:
                    penalty += base_weight
                else:
                    penalty += cardinal_ratio * base_weight

    return penalty, dropped_links

def compute_overlap_similarity(filtered_matches, cropped_patches):
    """
    Compute normalized cross-correlation (NCC) similarity between matching tile pairs
    using their cropped overlapping patches. Skips any non-overlapping or invalid pairs.

    Args:
        filtered_matches: Dict with key=(tile_a, tile_b), value={'direction': str, ...}
        cropped_patches: Dict[str, Dict[str, np.ndarray]], outer key is tile name, inner key is direction

    Returns:
        similarity_scores: Dict[(tile_a, tile_b), similarity_float]
    """
    import cv2
    import numpy as np

    similarity_scores = {}
    opposites = {'left': 'right', 'right': 'left', 'top': 'bottom', 'bottom': 'top'}

    for (tile_a, tile_b), match_data in filtered_matches.items():
        direction = match_data.get("direction")
        opposite = opposites.get(direction)

        patch_a_dict = cropped_patches.get(tile_a, {})
        patch_b_dict = cropped_patches.get(tile_b, {})

        patch_a = patch_a_dict.get(direction)
        patch_b = patch_b_dict.get(opposite)

        # Ensure valid NumPy arrays before proceeding
        if not isinstance(patch_a, np.ndarray) or not isinstance(patch_b, np.ndarray):
            continue

        # Ensure sizes match
        if patch_a.shape != patch_b.shape:
            continue

        # Compute normalized cross-correlation
        ncc = cv2.matchTemplate(patch_a, patch_b, cv2.TM_CCORR_NORMED)[0][0]
        similarity_scores[(tile_a, tile_b)] = float(ncc)

    return similarity_scores

def _count_candidate_links(neighbors: Dict[str, Dict[str, str]]) -> int:
    """
    Count unique undirected edges in a neighbor map:
      neighbors[tile_a][direction] = tile_b or None
    """
    seen = set()
    for a, dmap in neighbors.items():
        if not isinstance(dmap, dict):
            continue
        for _, b in dmap.items():
            if b is None:
                continue
            seen.add(tuple(sorted((a, b))))
    return len(seen)


# --- Add these near your imports / config ---
LARGE_FAIL_SCORE = 1e6  # finite but clearly bad; lets Optuna learn to avoid bad regions

def _compute_dataset_score(
    dataset_id: str,
    *,
    preloaded_by_id: Dict[str, Any],
    match_cache: Dict[Tuple[str, float, int], Any],
    lambda_reg: float,
    weight_method: str,
    ratio_thresh: float,
    min_matches: int,
    alpha: float,
    penalty_weight: float,
    cardinal_ratio: float,
    normalize_penalty_by_candidates: bool = True,
    similarity_as_diag: bool = True,
) -> Tuple[float, Dict[str, float]]:
    """
    Run the stitching pipeline for a single validation dataset and compute the scalar score.
    Now includes guards for degenerate 'no-match' situations.
    """
    # 1) Run pipeline on this dataset with the proposed hyperparameters
    x_corrected, x_noisy, A, b, filtered_results = run_stitching_pipeline_dataset_aware(
        dataset_id=dataset_id,
        preloaded_by_id=preloaded_by_id,
        match_cache=match_cache,
        lambda_reg=lambda_reg,
        weight_method=weight_method,
        ratio_thresh=ratio_thresh,
        min_matches=min_matches,
    )

    # 2) Fetch neighbor map and candidate link count for diagnostics/normalization
    neighbors = preloaded_by_id[dataset_id].neighbors
    candidate_links = _count_candidate_links(neighbors)

    # Guard A: no candidate links at all (degenerate dataset graph)
    if candidate_links == 0:
        details = {
            "constraint_error": 0.0,
            "displacement_error": 0.0,
            "penalty": 1.0,  # arbitrary unit penalty
            "alpha": float(alpha),
            "penalty_weight": float(penalty_weight),
            "cardinal_ratio": float(cardinal_ratio),
            "candidate_links": 0,
            "used_links": 0,
            "coverage": 0.0,
            "similarity_score": 0.0,
            "score": float(LARGE_FAIL_SCORE),
            "reason": "no_candidate_links",
        }
        return float(LARGE_FAIL_SCORE), details

    used_links = len(filtered_results) if hasattr(filtered_results, "__len__") else 0

    # Guard B: there were candidate links but *none* survived matching/filtering
    if used_links == 0:
        # Return a large finite score to strongly penalize this configuration.
        # We also set a maxed-out penalty (normalized) and zero coverage.
        penalty = 1.0  # normalized to candidate_links later, but we’re returning early
        score = float(LARGE_FAIL_SCORE)
        
        details = {
            "constraint_error": 0.0,
            "displacement_error": 0.0,
            "penalty": float(penalty),
            "alpha": float(alpha),
            "penalty_weight": float(penalty_weight),
            "cardinal_ratio": float(cardinal_ratio),
            "candidate_links": int(candidate_links),
            "used_links": 0,
            "coverage": 0.0,
            "similarity_score": 0.0,
            "score": float(score),
            "reason": "no_used_links",
        }
        return score, details

    # 3) Base score components from linear system + displacement regularization
    base_score, constraint_error, displacement_error = compute_score(
        x_corrected, x_noisy, A, b, alpha=alpha
    )

    # 4) Penalty for dropped/weak links (optionally normalized by candidate links)
    penalty, dropped_links = compute_link_dropout_penalty_ratio(
        neighbor_map=neighbors,
        filtered_matches=filtered_results,
        cardinal_ratio=cardinal_ratio,
        base_weight=1.0,
    )
    if normalize_penalty_by_candidates and candidate_links > 0:
        penalty = penalty / float(candidate_links)

    score = float(base_score) + float(penalty_weight) * float(penalty)

    # 5) Optional similarity diagnostic (not used in the objective)
    similarity_score = 0.0
    if similarity_as_diag:
        try:
            cropped_grayscale = getattr(preloaded_by_id[dataset_id], "overlaps", None)
            sim_map = compute_overlap_similarity(filtered_results, cropped_grayscale)
            similarity_score = float(np.mean(list(sim_map.values()))) if sim_map else 0.0
        except Exception:
            similarity_score = 0.0  # strictly diagnostic; ignore failures

    # 6) Collect details for logging
    details = {
        "constraint_error": float(constraint_error),
        "displacement_error": float(displacement_error),
        "penalty": float(penalty),
        "alpha": float(alpha),
        "penalty_weight": float(penalty_weight),
        "cardinal_ratio": float(cardinal_ratio),
        "candidate_links": int(candidate_links),
        "used_links": int(used_links),
        "coverage": float(used_links / candidate_links) if candidate_links > 0 else 0.0,
        "similarity_score": float(similarity_score),
        "score": float(score),
    }
    return score, details





# ----------------------------------------------------------------------
# Objective
# ----------------------------------------------------------------------
def objective(
    trial: optuna.trial.Trial,
    *,
    outer_fold: Dict[str, Any],
    inner_folds: List[Dict[str, Any]],
    preloaded_by_id: Dict[str, Any],
    match_cache: Dict[Tuple[str, float, int], Any],
    normalize_penalty_by_candidates: bool = True,
    similarity_as_diag: bool = True,
    map_none_to_uniform: bool = True,
) -> float:
    """
    Nested-CV objective for a single Optuna trial *within one OUTER fold*.

    Inputs:
        - outer_fold: dict with keys like 'fold_id', 'train_dataset_ids', 'test_dataset_ids', etc.
        - inner_folds: list of dicts; each has 'inner_id', 'inner_train_ids', 'inner_valid_ids'
        - preloaded_by_id: dataset_id -> PreloadedDataset
        - match_cache: shared cache across the entire run for (dataset_id, ratio_thresh, min_matches) matches
        - normalize_penalty_by_candidates: divide penalties by candidate link count (recommended)
        - similarity_as_diag: compute NCC similarity as a diagnostic only (not in the objective)
        - map_none_to_uniform: map weight_method 'none' to 'uniform' if your pipeline expects that name

    Returns:
        - mean_inner_score (float) across all inner folds (lower is better)
    """
    # 1) Sample hyperparameters (same ranges you proposed)
    lambda_reg = trial.suggest_float("lambda_reg", 0.01, 150.0, log=True)
    weight_method = trial.suggest_categorical(
        "weight_method", ["none", "raw_inliers", "inlier_ratio", "hybrid", "log_capped", "sqrt_inliers"]
    )
    ratio_thresh = trial.suggest_float("ratio_thresh", 0.6, 0.8)
    min_matches = trial.suggest_int("min_matches", 3, 10)
    alpha = trial.suggest_float("alpha", 1.0, 1000.0, log=True)
    penalty_weight = trial.suggest_float("penalty_weight", 0.1, 10.0, log=True)
    cardinal_ratio = trial.suggest_float("cardinal_ratio", 0.1, 3.0, log=True)

    # Optional mapping: some pipelines use "uniform" instead of "none"
    if map_none_to_uniform and weight_method == "none":
        weight_method = "uniform"

    # 2) Evaluate on INNER folds (validation on inner_valid_ids only)
    inner_scores: List[float] = []
    inner_details_all: Dict[str, Any] = {}
    try:
        for k, inner_fold in enumerate(inner_folds, start=1):
            valid_ids = inner_fold["inner_valid_ids"]
            fold_scores = []
            fold_details = {}

            for valid_id in valid_ids:
                s, d = _compute_dataset_score(
                    valid_id,
                    preloaded_by_id=preloaded_by_id,
                    match_cache=match_cache,
                    lambda_reg=lambda_reg,
                    weight_method=weight_method,
                    ratio_thresh=ratio_thresh,
                    min_matches=min_matches,
                    alpha=alpha,
                    penalty_weight=penalty_weight,
                    cardinal_ratio=cardinal_ratio,
                    normalize_penalty_by_candidates=normalize_penalty_by_candidates,
                    similarity_as_diag=similarity_as_diag,
                )
                fold_scores.append(s)
                fold_details[valid_id] = d

            inner_score = float(np.mean(fold_scores)) if fold_scores else float("inf")
            inner_scores.append(inner_score)
            inner_details_all[f"inner_{k}"] = fold_details

            # Report per-inner step for pruning
            trial.report(inner_score, step=k)
            if trial.should_prune():
                raise optuna.TrialPruned()

        mean_inner_score = float(np.mean(inner_scores)) if inner_scores else float("inf")

        # 3) Attach useful per-trial diagnostics
        trial.set_user_attr("outer_fold_id", int(outer_fold.get("fold_id", -1)))
        trial.set_user_attr("inner_scores", [float(s) for s in inner_scores])
        trial.set_user_attr("details", inner_details_all)
        trial.set_user_attr("params_effective", {
            "lambda_reg": float(lambda_reg),
            "weight_method": str(weight_method),
            "ratio_thresh": float(ratio_thresh),
            "min_matches": int(min_matches),
            "alpha": float(alpha),
            "penalty_weight": float(penalty_weight),
            "cardinal_ratio": float(cardinal_ratio),
            "normalize_penalty_by_candidates": bool(normalize_penalty_by_candidates),
        })

        return mean_inner_score

    except optuna.TrialPruned:
        raise  # Let Optuna handle pruning properly
    except Exception as e:
        # Fail-safe: return a large finite score so the search can continue
        trial.set_user_attr("failure", str(e))
        return float("1e12")


# ----------------------------------------------------------------------
# For Multi-Objective Scoring
# ----------------------------------------------------------------------

def compute_dimensionless_objectives(
    x_corrected,
    x_noisy,
    A,
    b,
    neighbor_map,
    tile_df,
    filtered_results,
    *,
    cardinal_ratio: float = 1.0,
    normalize_penalty_by_candidates: bool = True,
    assume_anchor_rows: int = 2,
    large_fail_score: float = 1e9,
):
    """
    Compute the three normalized (dimensionless/comparable) objectives in ONE call:
        (constraint_rms, disp_rms_norm, drop_rate), along with a diagnostics dict.
    """
    import numpy as np

    # ---------- inner helpers ----------
    def _slice_rule_rows(A_, b_, n_points_, anchor_rows_):
        total_rows = A_.shape[0]
        reg_rows = 2 * n_points_  # 2 rows per point regularizer (x and y)
        rule_rows = total_rows - reg_rows - anchor_rows_
        if rule_rows < 0:
            raise ValueError(
                f"Negative rule_rows={rule_rows}. "
                f"Check A shape ({A_.shape}) vs n_points={n_points_} and anchor_rows={anchor_rows_}."
            )
        return A_[:rule_rows, :], b_[:rule_rows]

    def _constraint_rms_unweighted(xcorr_, A_, b_, n_points_):
        A_rule, b_rule = _slice_rule_rows(A_, b_, n_points_, assume_anchor_rows)
        xvec = xcorr_.reshape(-1)
        resid = A_rule @ xvec - b_rule
        # De-weight each row by its max |coeff| to remove implicit row scaling
        row_scale = np.maximum(np.max(np.abs(A_rule), axis=1), 1e-12)
        resid_unweighted = resid / row_scale
        m = resid_unweighted.size
        return float(np.sqrt(np.sum(resid_unweighted**2) / max(m, 1)))

    def _median_grid_step(df_, neighbors_):
        coords = {str(r["tile_name"]): (float(r["x"]), float(r["y"])) for _, r in df_.iterrows()}
        cardinals = {"left", "right", "top", "bottom"}
        dists = []
        for a, nbrs in neighbors_.items():
            xa, ya = coords.get(str(a), (None, None))
            if xa is None:
                continue
            for direction, bname in (nbrs or {}).items():
                if bname is None:
                    continue
                if direction in cardinals:
                    xb, yb = coords.get(str(bname), (None, None))
                    if xb is None:
                        continue
                    d = np.hypot(xa - xb, ya - yb)
                    if d > 0:
                        dists.append(d)
        if not dists:
            for a, nbrs in neighbors_.items():
                xa, ya = coords.get(str(a), (None, None))
                if xa is None:
                    continue
                for _, bname in (nbrs or {}).items():
                    if bname is None:
                        continue
                    xb, yb = coords.get(str(bname), (None, None))
                    if xb is None:
                        continue
                    d = np.hypot(xa - xb, ya - yb)
                    if d > 0:
                        dists.append(d)
        return float(np.median(dists)) if dists else 0.0

    def _weighted_drop_rate(neighbors_, matches_, cardinal_ratio_):
        cardinals = {"left", "right", "top", "bottom"}
        C = 0.0  # weighted candidates
        D = 0.0  # weighted dropped
        used = 0.0
        dropped_links_list = []

        match_keys = set()
        if hasattr(matches_, "keys"):
            for k in matches_.keys():
                try:
                    a, b_ = k
                    match_keys.add((str(a), str(b_)))
                except Exception:
                    continue

        for a, nbrs in neighbors_.items():
            a = str(a)
            for direction, bname in (nbrs or {}).items():
                if bname is None:
                    continue
                b = str(bname)
                w = float(cardinal_ratio_) if direction in cardinals else 1.0
                C += w
                if (a, b) in match_keys or (b, a) in match_keys:
                    used += w
                else:
                    D += w
                    dropped_links_list.append((a, direction, b))

        coverage = (used / C) if C > 0 else 0.0
        drop = (D / C) if (normalize_penalty_by_candidates and C > 0) else D
        return float(drop), float(coverage), dropped_links_list, int(round(C))

    # ---------- quick guards ----------
    # Guard A: no candidate links in neighbor graph
    candidate_links_weighted = sum(
        1 for _, nbrs in (neighbor_map or {}).items() for v in (nbrs or {}).values() if v is not None
    )
    if candidate_links_weighted == 0:
        objectives = (large_fail_score, large_fail_score, large_fail_score)
        details = {
            "reason": "no_candidate_links",
            "candidate_links": 0,
            "used_links": 0,
            "coverage": 0.0,
            "similarity_score": 0.0,
            "n_dropped_links": 0,
            "n_unique_tiles_dropped": 0,
            "dropped_links": [],
            "disp_rms_raw_microns": 0.0,
            "grid_step_median": 0.0,
            "constraint_error": large_fail_score,
            "displacement_error": large_fail_score,
            "penalty": large_fail_score,
            "cardinal_ratio": float(cardinal_ratio),
        }
        return objectives, details

    # Guard B: candidates exist but no matches survived filtering
    if not filtered_results or (hasattr(filtered_results, "__len__") and len(filtered_results) == 0):
        objectives = (large_fail_score, large_fail_score, large_fail_score)
        details = {
            "reason": "no_used_links",
            "candidate_links": int(candidate_links_weighted),
            "used_links": 0,
            "coverage": 0.0,
            "similarity_score": 0.0,
            "n_dropped_links": 0,
            "n_unique_tiles_dropped": 0,
            "dropped_links": [],
            "disp_rms_raw_microns": 0.0,
            "grid_step_median": 0.0,
            "constraint_error": large_fail_score,
            "displacement_error": large_fail_score,
            "penalty": large_fail_score,
            "cardinal_ratio": float(cardinal_ratio),
        }
        return objectives, details

    # ---------- compute normalized objectives ----------
    n_points = int(x_corrected.shape[0])

    constraint_rms = _constraint_rms_unweighted(x_corrected, A, b, n_points)

    per_tile_disp = np.linalg.norm(np.asarray(x_corrected) - np.asarray(x_noisy), axis=1)
    disp_rms = float(np.sqrt(np.mean(per_tile_disp**2)))
    step_med = _median_grid_step(tile_df, neighbor_map)
    disp_rms_norm = disp_rms / max(step_med, 1e-12)

    drop_rate, coverage, dropped_links, weighted_candidates_est = _weighted_drop_rate(
        neighbor_map, filtered_results, cardinal_ratio
    )

    unique_tiles = set()
    for a, _, b_ in dropped_links:
        unique_tiles.add(a)
        unique_tiles.add(b_)

    used_links_est = int(round(coverage * weighted_candidates_est))

    objectives = (float(constraint_rms), float(disp_rms_norm), float(drop_rate))
    details = {
        "constraint_error": float(constraint_rms),
        "displacement_error": float(disp_rms_norm),
        "penalty": float(drop_rate),
        "cardinal_ratio": float(cardinal_ratio),
        "candidate_links": int(weighted_candidates_est),
        "used_links": int(used_links_est),
        "coverage": float(coverage),
        "similarity_score": 0.0,
        "n_dropped_links": int(len(dropped_links)),
        "n_unique_tiles_dropped": int(len(unique_tiles)),
        "dropped_links": dropped_links,
        "disp_rms_raw_microns": float(disp_rms),
        "grid_step_median": float(step_med),
    }
    return objectives, details


# def _multi_compute_dataset_score(
#     dataset_id: str,
#     *,
#     preloaded_by_id: Dict[str, Any],
#     match_cache: Dict[Tuple[str, float, int], Any],
#     lambda_reg: float,
#     weight_method: str,
#     ratio_thresh: float,
#     min_matches: int,
#     cardinal_ratio: float,
#     normalize_penalty_by_candidates: bool = True,
#     similarity_as_diag: bool = True,
# ) -> Tuple[Tuple[float, float, float], Dict[str, Any]]:
#     """
#     Run the stitching pipeline and compute the raw multi-objective scores for a single dataset.
#     Returns:
#         objectives: (constraint_error, displacement_error, penalty)  # all minimized
#         details:    diagnostics + full dropped links list
#     """
#     # 1) Run pipeline
#     x_corrected, x_noisy, A, b, filtered_results = run_stitching_pipeline_dataset_aware(
#         dataset_id=dataset_id,
#         preloaded_by_id=preloaded_by_id,
#         match_cache=match_cache,
#         lambda_reg=lambda_reg,
#         weight_method=weight_method,
#         ratio_thresh=ratio_thresh,
#         min_matches=min_matches,
#     )

#     # 2) Graph stats
#     neighbors = preloaded_by_id[dataset_id].neighbors
#     candidate_links = _count_candidate_links(neighbors)
#     used_links = len(filtered_results) if hasattr(filtered_results, "__len__") else 0

#     # --- Guard A: no candidate links at all ---
#     if candidate_links == 0:
#         print(f"[Guard A] {dataset_id}: no candidate links")
#         details = {
#             "constraint_error": 0.0,
#             "displacement_error": 0.0,
#             "penalty": 1.0,  # arbitrary penalty
#             "cardinal_ratio": float(cardinal_ratio),
#             "candidate_links": 0,
#             "used_links": 0,
#             "coverage": 0.0,
#             "similarity_score": 0.0,
#             "n_dropped_links": 0,
#             "n_unique_tiles_dropped": 0,
#             "dropped_links": [],
#             "reason": "no_candidate_links",
#         }
#         objectives = (LARGE_FAIL_SCORE, LARGE_FAIL_SCORE, LARGE_FAIL_SCORE)
#         return objectives, details

#     # --- Guard B: candidates exist but none survived matching/filtering ---
#     if used_links == 0:
#         print(f"[Guard B] {dataset_id}: {candidate_links} candidates but none survived")
#         details = {
#             "constraint_error": 0.0,
#             "displacement_error": 0.0,
#             "penalty": 1.0,  # normalized max penalty
#             "cardinal_ratio": float(cardinal_ratio),
#             "candidate_links": int(candidate_links),
#             "used_links": 0,
#             "coverage": 0.0,
#             "similarity_score": 0.0,
#             "n_dropped_links": 0,
#             "n_unique_tiles_dropped": 0,
#             "dropped_links": [],
#             "reason": "no_used_links",
#         }
#         objectives = (LARGE_FAIL_SCORE, LARGE_FAIL_SCORE, LARGE_FAIL_SCORE)
#         return objectives, details

#     # 3) Raw objective components (constraint, displacement, penalty)
#     _, constraint_obj, displacement_obj = compute_score(x_corrected, x_noisy, A, b, alpha=0.0)

#     # 4) Dropped-link penalty and stats
#     penalty_obj, dropped_links = compute_link_dropout_penalty_ratio(
#         neighbor_map=neighbors,
#         filtered_matches=filtered_results,
#         cardinal_ratio=cardinal_ratio,
#         base_weight=1.0,
#     )
#     if normalize_penalty_by_candidates and candidate_links > 0:
#         penalty_obj = penalty_obj / float(candidate_links)

#     # Counts + full dropped links list (converted for JSON storage)
#     n_dropped_links = int(len(dropped_links))
#     unique_tiles = set()
#     for a, _, b_ in dropped_links:
#         unique_tiles.add(a)
#         unique_tiles.add(b_)
#     n_unique_tiles_dropped = int(len(unique_tiles))
#     dropped_links_full = [list(t) for t in dropped_links]  # JSON-safe

#     # 5) Objective tuple
#     objectives = (float(constraint_obj), float(displacement_obj), float(penalty_obj))

#     # 6) Similarity diagnostic (optional)
#     similarity_score = 0.0
#     if similarity_as_diag:
#         try:
#             cropped_grayscale = getattr(preloaded_by_id[dataset_id], "overlaps", None)
#             sim_map = compute_overlap_similarity(filtered_results, cropped_grayscale)
#             similarity_score = float(np.mean(list(sim_map.values()))) if sim_map else 0.0
#         except Exception:
#             similarity_score = 0.0  # strictly diagnostic

#     # 7) Details for persistence
#     details = {
#         "constraint_error": float(constraint_obj),
#         "displacement_error": float(displacement_obj),
#         "penalty": float(penalty_obj),
#         "cardinal_ratio": float(cardinal_ratio),
#         "candidate_links": int(candidate_links),
#         "used_links": int(used_links),
#         "coverage": float(used_links / candidate_links) if candidate_links > 0 else 0.0,
#         "similarity_score": float(similarity_score),
#         "n_dropped_links": n_dropped_links,
#         "n_unique_tiles_dropped": n_unique_tiles_dropped,
#         "dropped_links": dropped_links_full,  # full list stored
#     }

#     # 8) Return
#     return objectives, details

def _multi_compute_dataset_score(
    dataset_id: str,
    *,
    preloaded_by_id: Dict[str, Any],
    match_cache: Dict[Tuple[str, float, int], Any],
    lambda_reg: float,
    weight_method: str,
    ratio_thresh: float,
    min_matches: int,
    cardinal_ratio: float,
    normalize_penalty_by_candidates: bool = True,
    similarity_as_diag: bool = True,
) -> Tuple[Tuple[float, float, float], Dict[str, Any]]:
    """
    Run the stitching pipeline and compute the **normalized** multi-objective scores for a single dataset.
    Returns:
        objectives: (constraint_rms_per_rule_eq, disp_rms_normalized, drop_rate)
        details:    diagnostics + full dropped links list
    """
    # 1) Run pipeline
    x_corrected, x_noisy, A, b, filtered_results = run_stitching_pipeline_dataset_aware(
        dataset_id=dataset_id,
        preloaded_by_id=preloaded_by_id,
        match_cache=match_cache,
        lambda_reg=lambda_reg,
        weight_method=weight_method,
        ratio_thresh=ratio_thresh,
        min_matches=min_matches,
    )

    ds = preloaded_by_id[dataset_id]

    # 2) Compute dimensionless objectives in one call
    objectives, details = compute_dimensionless_objectives(
        x_corrected=x_corrected,
        x_noisy=x_noisy,
        A=A,
        b=b,
        neighbor_map=ds.neighbors,
        tile_df=ds.tile_df,
        filtered_results=filtered_results,
        cardinal_ratio=cardinal_ratio,
        normalize_penalty_by_candidates=normalize_penalty_by_candidates,
        assume_anchor_rows=2,                 # set to 0 if you do NOT hard-anchor a tile
        large_fail_score=LARGE_FAIL_SCORE,
    )

    # 3) Optional similarity diagnostic (kept identical to your previous behavior)
    if similarity_as_diag:
        try:
            cropped_grayscale = getattr(ds, "overlaps", None)
            sim_map = compute_overlap_similarity(filtered_results, cropped_grayscale)
            details["similarity_score"] = float(np.mean(list(sim_map.values()))) if sim_map else 0.0
        except Exception:
            details["similarity_score"] = 0.0

    return objectives, details

def multi_objective(
    trial: optuna.trial.Trial,
    *,
    outer_fold: Dict[str, Any],
    inner_folds: List[Dict[str, Any]],
    preloaded_by_id: Dict[str, Any],
    match_cache: Dict[Tuple[str, float, int], Any],
    normalize_penalty_by_candidates: bool = True,
    similarity_as_diag: bool = True,
    map_none_to_uniform: bool = True,
) -> Tuple[float, float, float]:
    """
    Nested-CV multi-objective function for a single Optuna trial.
    Returns:
        mean_objectives: (constraint_error, displacement_error, penalty), averaged across inner folds.
    """

    # 1) Sample hyperparameters
    # lambda_reg = trial.suggest_float("lambda_reg", 0.01, 150.0, log=True)    
    # weight_method = trial.suggest_categorical(
    #     "weight_method", ["none", "raw_inliers", "inlier_ratio", "hybrid", "log_capped"] # Got rid of Raw Inliers during primary search
    # )
    # ratio_thresh = trial.suggest_float("ratio_thresh", 0.6, 0.8)
    # min_matches = trial.suggest_int("min_matches", 3, 10)
    # cardinal_ratio = trial.suggest_float("cardinal_ratio", 0.1, 3.0, log=True)

    # 1) Sample hyperparameters (tightened to cross-fold q10–q90 intersections)
    lambda_reg = trial.suggest_float("lambda_reg", 0.061, 53.445, log=True)
    weight_method = trial.suggest_categorical(
        "weight_method", ["inlier_ratio", "hybrid", "none", "log_capped"]  
    )
    ratio_thresh = trial.suggest_float("ratio_thresh", 0.624, 0.781)
    min_matches = trial.suggest_int("min_matches", 5, 9)
    cardinal_ratio = trial.suggest_float("cardinal_ratio", 0.238, 1.196, log=True)
    if map_none_to_uniform and weight_method == "none":
        weight_method = "uniform"

    # 2) Evaluate on INNER folds
    inner_fold_objectives: List[Tuple[float, float, float]] = []
    inner_details_all: Dict[str, Any] = {}

    try:
        # accumulators for rollups
        all_coverages = []
        all_similarities = []
        all_dropped_counts = []
        all_unique_tile_counts = []
        all_dropped_links = []  # optional, full concatenation across datasets

        for k, inner_fold in enumerate(inner_folds, start=1):
            valid_ids = inner_fold["inner_valid_ids"]
            current_fold_objectives = []
            fold_details = {}

            for valid_id in valid_ids:
                objectives_tuple, details = _multi_compute_dataset_score(
                    dataset_id=valid_id,
                    preloaded_by_id=preloaded_by_id,
                    match_cache=match_cache,
                    lambda_reg=lambda_reg,
                    weight_method=weight_method,
                    ratio_thresh=ratio_thresh,
                    min_matches=min_matches,
                    cardinal_ratio=cardinal_ratio,
                    normalize_penalty_by_candidates=normalize_penalty_by_candidates,
                    similarity_as_diag=similarity_as_diag,
                )
                current_fold_objectives.append(objectives_tuple)
                fold_details[valid_id] = details

                # accumulate diagnostics
                all_coverages.append(details["coverage"])
                all_similarities.append(details["similarity_score"])
                all_dropped_counts.append(details["n_dropped_links"])
                all_unique_tile_counts.append(details["n_unique_tiles_dropped"])
                all_dropped_links.extend(details["dropped_links"])

            mean_objectives_for_fold = tuple(np.mean(current_fold_objectives, axis=0))
            inner_fold_objectives.append(mean_objectives_for_fold)
            inner_details_all[f"inner_{k}"] = fold_details

        # Average each objective independently across inner folds
        mean_objectives = tuple(np.mean(inner_fold_objectives, axis=0))

        # 3) Attach trial-level diagnostics
        trial.set_user_attr("outer_fold_id", int(outer_fold.get("fold_id", -1)))
        trial.set_user_attr("details", inner_details_all)

        # aggregated diagnostics
        trial.set_user_attr("coverage_mean", float(np.mean(all_coverages)) if all_coverages else 0.0)
        trial.set_user_attr("coverage_min", float(np.min(all_coverages)) if all_coverages else 0.0)
        trial.set_user_attr("similarity_mean", float(np.mean(all_similarities)) if all_similarities else 0.0)
        trial.set_user_attr("n_dropped_links_mean", float(np.mean(all_dropped_counts)) if all_dropped_counts else 0.0)
        trial.set_user_attr("n_dropped_links_max", int(np.max(all_dropped_counts)) if all_dropped_counts else 0)
        trial.set_user_attr("n_unique_tiles_dropped_mean", float(np.mean(all_unique_tile_counts)) if all_unique_tile_counts else 0.0)

        # optional: store full concatenated dropped links for inspection
        trial.set_user_attr("all_dropped_links", all_dropped_links)

        # effective params (self-contained record)
        trial.set_user_attr("params_effective", {
            "lambda_reg": float(lambda_reg),
            "weight_method": str(weight_method),
            "ratio_thresh": float(ratio_thresh),
            "min_matches": int(min_matches),
            "cardinal_ratio": float(cardinal_ratio),
            "normalize_penalty_by_candidates": bool(normalize_penalty_by_candidates),
        })

        return mean_objectives

    except Exception as e:
        import traceback
        traceback.print_exc()
        trial.set_user_attr("failure", str(e))
        return (1e12, 1e12, 1e12)



if __name__ == "__main__":
    # ==================================================================
    # SECTION 1-3: DATA LOADING AND FOLD SETUP
    # ==================================================================

    # 1) Build registry + preload (as before)
    registry = build_fixed_registry(DATASETS_ROOT)

    preloaded_by_id = preload_all_for_cv(
        registry,
        tolerance=0.25,
        dbscan_eps=2000.0,
        dbscan_min_samples=2,
        overlap_fraction=0.25,
        extra_factor=0.10
    )

    # 2) Warm up cache with a single per-dataset run (optional but helps speed)
    match_cache = {}
    _lambda_reg   = 1.0
    _weight_method = "uniform"
    _ratio_thresh = 0.75
    _min_matches  = 8

    print("\n[Sanity] Warming cache with one pass:")
    for dataset_id in preloaded_by_id.keys():
        corrected_points, noisy_points, A, b, filtered_results = run_stitching_pipeline_dataset_aware(
            dataset_id=dataset_id,
            preloaded_by_id=preloaded_by_id,
            match_cache=match_cache,
            lambda_reg=_lambda_reg,
            weight_method=_weight_method,
            ratio_thresh=_ratio_thresh,
            min_matches=_min_matches
        )
        n_links = len(filtered_results) if hasattr(filtered_results, "__len__") else "unknown"
        print(f"  -> {dataset_id}: links_used={n_links}")

    # 3) Build OUTER folds (LODO) and pick ONE fold to smoke-test the objective
    print("\n[CV] Building OUTER folds (LODO)...")
    outer_folds = build_outer_folds(registry, preloaded_by_id)

    # Choose the first outer fold for a quick objective test
    outer = outer_folds[0]
    inner_folds = build_inner_folds(outer["train_dataset_ids"], registry)


    # # ==================================================================
    # # SECTION 4-5: SINGLE-OBJECTIVE SMOKE TEST (UNCHANGED)
    # # ==================================================================
    # print("\n\n" + "="*50)
    # print("--- RUNNING SINGLE-OBJECTIVE SMOKE TEST ---")
    # print("="*50)

    # # 4) Create a tiny Optuna study to test the objective wiring
    # N_TRIALS = 5  # keep this small for a quick smoke test
    # sampler = TPESampler(seed=123)
    # pruner  = MedianPruner()  # can set to None to disable
    # study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner, study_name="objective_smoke_test")

    # print(f"\n[Optuna] Running {N_TRIALS} trial(s) on OUTER fold {outer['fold_id']}...")
    # study.optimize(
    #     lambda trial: objective(
    #         trial,
    #         outer_fold=outer,
    #         inner_folds=inner_folds,
    #         preloaded_by_id=preloaded_by_id,
    #         match_cache=match_cache,
    #         normalize_penalty_by_candidates=True,
    #         similarity_as_diag=True,
    #         map_none_to_uniform=True,
    #     ),
    #     n_trials=N_TRIALS,
    #     show_progress_bar=False,
    # )

    # # 5) Show best trial and evaluate once on the OUTER test dataset
    # best = study.best_trial
    # params = best.params.copy()
    # if params.get("weight_method") == "none":
    #     params["weight_method"] = "uniform"

    # print(f"\n[Result] Best inner-CV score = {best.value:.6f}")
    # print(f"[Result] Best params = {params}")
    # print(f"[Result] Inner fold scores = {best.user_attrs.get('inner_scores', [])}")

    test_ids = outer["test_dataset_ids"]
    # print(f"\n[Hold-out] Evaluating best params on OUTER test set {test_ids}...")
    # test_scores = []
    # for test_id in test_ids:
    #     score, details = _compute_dataset_score(
    #         test_id,
    #         preloaded_by_id=preloaded_by_id,
    #         match_cache=match_cache,
    #         lambda_reg=float(params["lambda_reg"]),
    #         weight_method=str(params["weight_method"]),
    #         ratio_thresh=float(params["ratio_thresh"]),
    #         min_matches=int(params["min_matches"]),
    #         alpha=float(params["alpha"]),
    #         penalty_weight=float(params["penalty_weight"]),
    #         cardinal_ratio=float(params["cardinal_ratio"]),
    #         normalize_penalty_by_candidates=True,
    #         similarity_as_diag=True,
    #     )
    #     test_scores.append(score)
    #     print(f"  - {test_id}: score={details['score']:.6f}  coverage={details['coverage']:.3f}  "
    #           f"constraint={details['constraint_error']:.4f}  disp={details['displacement_error']:.4f}  "
    #           f"penalty={details['penalty']:.4f}")

    # mean_test = float(np.mean(test_scores)) if test_scores else float("inf")
    # print(f"\n[Summary] OUTER fold {outer['fold_id']} mean TEST score = {mean_test:.6f}")


    # ==================================================================
    # SECTION 6: NEW MULTI-OBJECTIVE SMOKE TEST (EXPANSION)
    # ==================================================================
    print("\n\n" + "="*50)
    print("--- RUNNING MULTI-OBJECTIVE SMOKE TEST ---")
    print("="*50)

    # 6.1) Create a new Optuna study for Multi-Objective Optimization (MOO)
    N_TRIALS_MO = 10 # We can run a few more trials for MOO to see the front develop
    
    # A specialized sampler like NSGAIISampler is highly recommended for MOO
    mo_sampler = optuna.samplers.NSGAIISampler(seed=123)
    mo_pruner = MedianPruner()

    # The key change: specify `directions` (plural) for each objective.
    mo_study = optuna.create_study(
        directions=["minimize", "minimize", "minimize"], # One for each objective in your tuple
        study_name="multi_objective_smoke_test",
        sampler=mo_sampler,
        pruner=mo_pruner
    )

    print(f"\n[Optuna-MO] Running {N_TRIALS_MO} trial(s) on OUTER fold {outer['fold_id']}...")
    mo_study.optimize(
        lambda trial: multi_objective( 
            trial,
            outer_fold=outer,
            inner_folds=inner_folds,
            preloaded_by_id=preloaded_by_id,
            match_cache=match_cache,
            normalize_penalty_by_candidates=True,
            similarity_as_diag=True,
            map_none_to_uniform=True,
        ),
        n_trials=N_TRIALS_MO,
        show_progress_bar=False,
    )

    # 6.2) Show the results. For MOO, we look at the "best_trials" (plural) on the Pareto front.
    print(f"\n[MO-Result] Found {len(mo_study.best_trials)} non-dominated solution(s) on the Pareto front.")
    
    # Sort trials by the first objective (constraint_error) for cleaner display
    sorted_pareto_front = sorted(mo_study.best_trials, key=lambda t: t.values[0])

    print("\n[MO-Pareto Front Summary]")
    print(f"{'Trial':<6} | {'Constraint':>12} | {'Displacement':>12} | {'Penalty':>10} | Params")
    print("-" * 80)
    for trial in sorted_pareto_front:
        c, d, p = trial.values
        print(f"{trial.number:<6} | {c:12.6f} | {d:12.6f} | {p:10.6f} | {trial.params}")

    if mo_study.best_trials:
        best_by_constraint = min(mo_study.best_trials, key=lambda t: t.values[0])
        mo_params = best_by_constraint.params.copy()
        if mo_params.get("weight_method") == "none":
            mo_params["weight_method"] = "uniform"

        print(f"\n[Hold-out] Evaluating Pareto solution with lowest constraint error (Trial {best_by_constraint.number}) on {test_ids}...")
        
        mo_test_objectives = []
        for test_id in test_ids:
            # We use the multi-objective scoring function for evaluation
            objectives, details = _multi_compute_dataset_score(
                test_id,
                preloaded_by_id=preloaded_by_id,
                match_cache=match_cache,
                lambda_reg=float(mo_params["lambda_reg"]),
                weight_method=str(mo_params["weight_method"]),
                ratio_thresh=float(mo_params["ratio_thresh"]),
                min_matches=int(mo_params["min_matches"]),
                cardinal_ratio=float(mo_params["cardinal_ratio"]),
                normalize_penalty_by_candidates=True,
                similarity_as_diag=True,
            )
            mo_test_objectives.append(objectives)
            
            # --- NEW: Convert tuple for clean printing ---
            # This line converts each NumPy number to a standard Python float,
            # formatted to 6 decimal places for a tidy appearance.
            clean_objectives_for_print = tuple(float(f"{x:.6f}") for x in objectives)
            print(f"  - {test_id}: Objectives={clean_objectives_for_print} | coverage={details['coverage']:.3f}")
        
        # Calculate the mean of the raw objective values
        mean_test_objectives = tuple(np.mean(mo_test_objectives, axis=0))

        clean_mean_objectives_for_print = tuple(float(f"{x:.6f}") for x in mean_test_objectives)
        print(f"\n[Summary] OUTER fold {outer['fold_id']} mean TEST objectives = {clean_mean_objectives_for_print}")

