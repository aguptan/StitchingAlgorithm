import sys
from typing import Dict, List
from collections import Counter

from DatasetLoading import (
    build_fixed_registry,
    preload_all_for_cv,
    run_stitching_pipeline_dataset_aware,
)

DATASETS_ROOT = r"C:\Users\agupt\Desktop\Shoykhet_Lab\Stitching\Optimization\OptimizationDataSet"
FUNCTIONS_DIR = r"C:\Users\agupt\Desktop\Shoykhet_Lab\Stitching\Algorithm\Functions"

if FUNCTIONS_DIR not in sys.path:
    sys.path.insert(0, FUNCTIONS_DIR)

# ----------------------------------
# 2. Building Cross Validation Folds
# ----------------------------------
# NOTE: This expects your project types:
# - DatasetEntry: has fields .dataset_id (str) and .domain (str)
# - PreloadedDataset: has field .neighbors (dict) mapping tile -> direction -> neighbor_tile (or None)

def build_outer_folds(
    registry: List["DatasetEntry"],
    preloaded_by_id: Dict[str, "PreloadedDataset"],
) -> List[dict]:
    """
    Leave-One-Dataset-Out (LODO) OUTER folds.

    Returns a list of dicts with:
      - fold_id
      - train_dataset_ids
      - test_dataset_ids
      - train_domain_counts
      - test_domain_counts
      - domain_shift_L1
      - candidate_links_train
      - candidate_links_test
      - mode_used = "LODO-DS"
    """
    # Stable dataset list limited to preloaded
    reg_map = {e.dataset_id: e for e in registry}
    all_ids = sorted(dsid for dsid in preloaded_by_id.keys() if dsid in reg_map)
    if len(all_ids) < 2:
        raise ValueError("Need at least 2 preloaded datasets for outer CV (LODO).")

    domain_of = lambda dsid: reg_map[dsid].domain

    def count_links(neighbors: dict) -> int:
        seen = set()
        for a, dmap in neighbors.items():
            if not isinstance(dmap, dict):
                continue
            for _, b in dmap.items():
                if b is None:
                    continue
                seen.add(tuple(sorted((a, b))))
        return len(seen)

    def domain_counts(ids: List[str]) -> Dict[str, int]:
        return dict(Counter(domain_of(i) for i in ids))

    def l1_shift(train_ids: List[str], test_ids: List[str]) -> float:
        tc, vc = domain_counts(train_ids), domain_counts(test_ids)
        keys = set(tc) | set(vc)
        tot_t, tot_v = max(sum(tc.values(), 0), 1), max(sum(vc.values(), 0), 1)
        return sum(abs(tc.get(k, 0) / tot_t - vc.get(k, 0) / tot_v) for k in keys)

    folds: List[dict] = []
    for fid, held_out in enumerate(all_ids, start=1):
        test_ids = [held_out]
        train_ids = [i for i in all_ids if i != held_out]
        folds.append({
            "fold_id": fid,
            "train_dataset_ids": train_ids,
            "test_dataset_ids": test_ids,
            "train_domain_counts": domain_counts(train_ids),
            "test_domain_counts": domain_counts(test_ids),
            "domain_shift_L1": l1_shift(train_ids, test_ids),
            "candidate_links_train": sum(count_links(preloaded_by_id[i].neighbors) for i in train_ids),
            "candidate_links_test": sum(count_links(preloaded_by_id[i].neighbors) for i in test_ids),
            "mode_used": "LODO-DS",
        })

    return folds

def build_inner_folds(
    train_dataset_ids: List[str],
    registry: List["DatasetEntry"],
) -> List[dict]:
    """
    Leave-One-Dataset-Out (LODO) INNER folds over the OUTER-TRAIN pool.

    Returns a list of dicts with:
      - inner_id
      - inner_train_ids
      - inner_valid_ids
      - mode_used = "LODO-DS"
    """
    reg_map = {e.dataset_id: e for e in registry}
    ids = sorted(i for i in train_dataset_ids if i in reg_map)
    if len(ids) < 2:
        raise ValueError("Need at least 2 datasets in the outer-train pool for inner CV (LODO).")

    folds: List[dict] = []
    for k, held_out in enumerate(ids, start=1):
        inner_train = [i for i in ids if i != held_out]
        folds.append({
            "inner_id": k,
            "inner_train_ids": inner_train,
            "inner_valid_ids": [held_out],
            "mode_used": "LODO-DS",
        })

    return folds


if __name__ == "__main__":
    
    registry = build_fixed_registry(DATASETS_ROOT)

    preloaded_by_id = preload_all_for_cv(
        registry,
        tolerance=0.25,
        dbscan_eps=2000.0,
        dbscan_min_samples=2,
        overlap_fraction=0.25,
        extra_factor=0.10
    )

    # 2) Basic per-dataset run 
    match_cache = {}
    lambda_reg   = 1.0
    weight_method = "uniform"
    ratio_thresh = 0.75
    min_matches  = 8

    # Build OUTER folds (LODO) and display diagnostics
    print("\n[CV] Building OUTER folds (LODO)...")
    outer_folds = build_outer_folds(registry, preloaded_by_id)
    for f in outer_folds:
        print(f"  [Outer {f['fold_id']}] mode={f['mode_used']}")
        print(f"    train={f['train_dataset_ids']}")
        print(f"    test ={f['test_dataset_ids']}")
        print(f"    train_domain_counts={f['train_domain_counts']}  test_domain_counts={f['test_domain_counts']}")
        print(f"    domain_shift_L1={f['domain_shift_L1']:.4f}")
        print(f"    candidate_links_train={f['candidate_links_train']}  candidate_links_test={f['candidate_links_test']}")

    # For each OUTER fold, build INNER folds (LODO on the train pool) and dry-run the pipeline on the inner VALID sets
    print("\n[CV] Building INNER folds (LODO) per OUTER fold and dry-running validation datasets...")
    for f in outer_folds:
        train_ids = f["train_dataset_ids"]
        test_ids  = f["test_dataset_ids"]
        # Safety check: disjoint sets
        assert set(train_ids).isdisjoint(set(test_ids)), "Outer train/test overlap detected."

        inner_folds = build_inner_folds(train_ids, registry)
        print(f"\n  [Outer {f['fold_id']}] INNER folds:")
        for g in inner_folds:
            print(f"    (Inner {g['inner_id']}) mode={g['mode_used']}")
            print(f"      inner_train={g['inner_train_ids']}")
            print(f"      inner_valid={g['inner_valid_ids']}")

            # Dry-run the stitching pipeline on the inner VALID dataset(s)
            for valid_id in g["inner_valid_ids"]:
                print(f"        -> Dry-run valid dataset: {valid_id}")
                corrected_points, noisy_points, A, b, filtered_results = run_stitching_pipeline_dataset_aware(
                    dataset_id=valid_id,
                    preloaded_by_id=preloaded_by_id,
                    match_cache=match_cache,
                    lambda_reg=lambda_reg,
                    weight_method=weight_method,
                    ratio_thresh=ratio_thresh,
                    min_matches=min_matches
                )
                n_links = len(filtered_results) if hasattr(filtered_results, "__len__") else "unknown"
                print(f"           links_used={n_links}, corrected_points={getattr(corrected_points, 'shape', None)}, noisy_points={getattr(noisy_points, 'shape', None)}")

        # Quick dry-run on OUTER test set, using same hyperparameters
        print(f"\n  [Outer {f['fold_id']}] Dry-run on OUTER test set:")
        for test_id in test_ids:
            print(f"    -> Test dataset: {test_id}")
            corrected_points, noisy_points, A, b, filtered_results = run_stitching_pipeline_dataset_aware(
                dataset_id=test_id,
                preloaded_by_id=preloaded_by_id,
                match_cache=match_cache,
                lambda_reg=lambda_reg,
                weight_method=weight_method,
                ratio_thresh=ratio_thresh,
                min_matches=min_matches
            )
            n_links = len(filtered_results) if hasattr(filtered_results, "__len__") else "unknown"
            print(f"       links_used={n_links}, corrected_points={getattr(corrected_points, 'shape', None)}, noisy_points={getattr(noisy_points, 'shape', None)}")

    print("\n[Done] CV dry-run complete. You can now plug in the objective and Optuna loop.")