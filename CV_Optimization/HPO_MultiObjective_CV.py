# ======================================================================
# Main Experiment Orchestrator Script
# ======================================================================
"""
This script orchestrates the full nested cross-validation experiment for
multi-objective hyperparameter optimization of the stitching algorithm.

It follows a structured, multi-phase process:
1.  **Global Setup**: Imports, configuration, and one-time data loading.
2.  **Outer CV Loop**: Iterates through each primary fold, where one dataset
    is held out as the final test set.
3.  **Inner HPO Loop**: For each outer fold, it runs a full, independent
    multi-objective Optuna study to find the best hyperparameters using
    only the training/validation data for that fold.
4.  **Result Saving**: Saves the complete Optuna study object for each
    outer fold, allowing for detailed post-hoc analysis in a separate script.
"""

# --- Standard Library Imports ---
import os
import sys
import joblib
from datetime import datetime
import numpy as np, random

# --- Third-Party Library Imports ---
import optuna
GLOBAL_SEED = 123
np.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)
try:
    import torch
    torch.manual_seed(GLOBAL_SEED)
    torch.cuda.manual_seed_all(GLOBAL_SEED)
except ImportError:
    pass

print(f"[Setup] Global seed set to {GLOBAL_SEED}")

N_JOBS = 6

# --- Local Module Imports ---
# This script assumes it is run from a location where it can find the 'Modules'
# directory, or the path is configured correctly.
MODULES_DIR = r"C:\Users\agupt\Desktop\Shoykhet_Lab\Stitching\Optimization\Modules" #<-- UPDATE IF NEEDED
if MODULES_DIR not in sys.path:
    sys.path.insert(0, MODULES_DIR)

from DatasetLoading import build_fixed_registry, preload_all_for_cv
from CrossValidationTesting import build_outer_folds, build_inner_folds
from ObjectiveCreation import multi_objective

# ======================================================================
# Phase 1: Global Setup and Configuration
# ======================================================================

# --- Top-Level Configuration ---
DATASETS_ROOT = r"C:\Users\agupt\Desktop\Shoykhet_Lab\Stitching\Optimization\OptimizationDataSet" 

# Directory to save all experimental results (Optuna .db and final .pkl files)
RESULTS_DIR = r"C:\Users\agupt\Desktop\Shoykhet_Lab\Stitching\Optimization\optuna_results\Multi_Objective"

# Number of hyperparameter optimization trials to run for EACH outer fold
N_TRIALS_PER_FOLD = 492

# A unique prefix for this experimental run, using a timestamp
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
STUDY_NAME_PREFIX = f"stitching_moo_{RUN_TIMESTAMP}"


def main():
    """Main function to run the entire experiment."""
    print("=" * 70)    
    print("Stitching Algorithm: Multi-Objective Hyperparameter Optimization")
    print("=" * 70)
    print(f"Timestamp for this run: {RUN_TIMESTAMP}")
    print(f"Number of Optuna trials per fold: {N_TRIALS_PER_FOLD}")
    print(f"Results will be saved in: {RESULTS_DIR}")
    print("-" * 70)

    # Ensure the results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- One-Time Data Loading ---
    print("\n[Phase 1] Preloading all datasets into memory... (This may take a moment)")
    try:
        registry = build_fixed_registry(DATASETS_ROOT)
        preloaded_by_id = preload_all_for_cv(
            registry,
            tolerance=0.25,
            dbscan_eps=2000.0,
            dbscan_min_samples=2,
            overlap_fraction=0.25,
            extra_factor=0.10
        )
        print(f"Successfully preloaded {len(preloaded_by_id)} datasets.")
    except Exception as e:
        print(f"\n[FATAL] Failed to preload data: {e}")
        print("Please check the DATASETS_ROOT path and dataset integrity.")
        return # Exit if data loading fails

    # ======================================================================
    # Phase 2: The Main Experimental Loop (Outer Cross-Validation)
    # ======================================================================
    print("\n[Phase 2] Building outer cross-validation folds (Leave-One-Dataset-Out)...")
    outer_folds = build_outer_folds(registry, preloaded_by_id)
    n_outer_folds = len(outer_folds)
    print(f"Successfully created {n_outer_folds} outer folds.")

    # --- Loop Through Folds ---
    for i, outer_fold in enumerate(outer_folds, start=1):
        fold_id = outer_fold['fold_id']
        test_dataset_id = outer_fold['test_dataset_ids'][0] # LODO means one test set

        print("\n" + "=" * 70)
        print(f"Starting Outer Fold {i}/{n_outer_folds} (ID: {fold_id})")
        print(f"  - Test Set (Held Out): {test_dataset_id}")
        print(f"  - Training Sets: {outer_fold['train_dataset_ids']}")
        print("=" * 70)

        # ======================================================================
        # Phase 3: Per-Fold Hyperparameter Optimization (Inside the Main Loop)
        # ======================================================================

        # --- Fold-Specific Setup ---
        study_name = f"{STUDY_NAME_PREFIX}_fold_{fold_id}"
        storage_path = f"sqlite:///{os.path.join(RESULTS_DIR, f'{study_name}.db')}?timeout=60"
        final_study_path = os.path.join(RESULTS_DIR, f"{study_name}_final.pkl")
        
        print(f"[Phase 3] Preparing for HPO for Fold {fold_id}...")
        print(f"  - Study Name: {study_name}")
        print(f"  - DB Storage: {storage_path}")

        # Build the inner validation folds using only this outer fold's training data
        inner_folds = build_inner_folds(outer_fold["train_dataset_ids"], registry)
        print(f"  - Built {len(inner_folds)} inner folds for validation.")

        # Create a fresh match_cache for each outer fold to ensure independence
        # Thread-local match_cache per worker to avoid write races with n_jobs>1
        import threading
        _thread_local = threading.local()
        def _get_match_cache():
            if not hasattr(_thread_local, "match_cache"):
                _thread_local.match_cache = {}
            return _thread_local.match_cache


        # --- Create an Independent Study ---
        # A specialized sampler like NSGAIISampler is best for multi-objective optimization.
        sampler = optuna.samplers.NSGAIISampler(seed=GLOBAL_SEED)
        pruner = optuna.pruners.MedianPruner()

        study = optuna.create_study(
            study_name=study_name,
            storage=storage_path,
            directions=["minimize", "minimize", "minimize"], # For the 3 objectives
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True # Allows the script to be resumed if it fails
        )
        
        # Trial outcomes summary
        completed = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
        pruned    = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
        failed    = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.FAIL)
        running   = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.RUNNING)

        study.set_user_attr("trial_outcomes", {
            "completed": int(completed),
            "pruned": int(pruned),
            "failed": int(failed),
            "running": int(running),
            "pareto_size": int(len(study.best_trials)),
        })
        study.set_user_attr("preprocess_config", {
            "tolerance": 0.25,
            "dbscan_eps": 2000.0,
            "dbscan_min_samples": 2,
            "overlap_fraction": 0.25,
            "extra_factor": 0.10,
        })
        
        outer_meta = {
            "fold_id": fold_id,
            "train_ids": outer_fold["train_dataset_ids"],
            "test_ids": outer_fold["test_dataset_ids"],
            "n_train": len(outer_fold["train_dataset_ids"]),
            "n_test": len(outer_fold["test_dataset_ids"]),
            "domain_shift_L1": outer_fold.get("domain_shift_L1"),

        }
        study.set_user_attr("outer_fold", outer_meta)
        study.set_user_attr("objective_names", ["constraint_rms", "disp_rms_norm", "drop_rate"])
        study.set_user_attr(
                            "objective_notes",
                            "constraint_rms = RMS over unweighted rule rows; "
                            "disp_rms_norm = RMS displacement / median grid step; "
                            "drop_rate = weighted dropped / weighted candidates."
                        )
        study.set_user_attr("normalization_config", {
                                "penalty_normalized_by_candidates": True,
                                "assume_anchor_rows": 2,  # set to 0 if you later drop the hard anchor
                                "constraint_metric": "rms_per_rule_eq_unweighted",
                                "displacement_metric": "rms_over_median_grid_step",
                                "penalty_metric": "weighted_drop_rate",
                            })
        
        study.set_user_attr("global_config", {
            "timestamp": RUN_TIMESTAMP,
            "seed": GLOBAL_SEED,
            "datasets_root": DATASETS_ROOT,
            "results_dir": RESULTS_DIR,
            "n_trials_per_fold": N_TRIALS_PER_FOLD,
            "study_prefix": STUDY_NAME_PREFIX,
            "n_jobs": int(N_JOBS)
        })
        
        study.set_user_attr("inner_folds", {
                i+1: {
                    "train_ids": inner["inner_train_ids"],
                    "valid_ids": inner["inner_valid_ids"],
                }
                for i, inner in enumerate(inner_folds)
            })
        # --- Run the Optimization ---
        print(f"\n[Optuna] Starting optimization with {N_TRIALS_PER_FOLD} trials...")
        # The 'test_dataset_ids' are part of 'outer_fold' but are NEVER used by
        # the 'multi_objective' function, which correctly uses 'inner_folds' for validation.
        
        study.optimize(
            lambda trial: multi_objective(
                trial,
                outer_fold=outer_fold,  # for logging/attrs only
                inner_folds=inner_folds,
                preloaded_by_id=preloaded_by_id,
                match_cache=_get_match_cache(),          # thread-local cache
                normalize_penalty_by_candidates=True,    # lock defaults explicitly
                similarity_as_diag=True,
                map_none_to_uniform=True,
            ),
            n_trials=N_TRIALS_PER_FOLD,
            n_jobs=N_JOBS,                               # <<< Optuna multithreading here
            show_progress_bar=False,
        )

        pareto_summary = [
                {
                    "trial_number": t.number,
                    "values": tuple(float(v) for v in t.values),
                    "params": t.params
                }
                for t in study.best_trials
            ]
        study.set_user_attr("pareto_front", pareto_summary)
        all_coverages = [t.user_attrs.get("coverage_mean", 0.0) for t in study.trials if t.values]
        study.set_user_attr("coverage_summary", {
            "mean": float(np.mean(all_coverages)),
            "min": float(np.min(all_coverages)),
            "max": float(np.max(all_coverages)),
        })

        print(f"\n[Optuna] Optimization complete for Fold {fold_id}.")
        print(f"  - Found {len(study.best_trials)} non-dominated solutions (Pareto front).")

        # --- Save the Final Study Object ---
        print(f"  - Saving final study object to: {final_study_path}")
        try:
            joblib.dump(study, final_study_path)
            print("  - Save successful.")
        except Exception as e:
            print(f"[ERROR] Could not save the final study object for Fold {fold_id}: {e}")


    # ======================================================================
    # Phase 4: Experiment Completion
    # ======================================================================
    print("\n" + "=" * 70)
    print("[Phase 4] Experiment Complete!")
    print("=" * 70)
    print("All outer folds have been processed.")
    print(f"The results (one .db and one .pkl file per fold) are saved in:\n{RESULTS_DIR}")
    print("\nYou can now proceed with the analysis script to evaluate the Pareto fronts")
    print("and assess generalization performance on the held-out test sets.")


if __name__ == "__main__":
    main()