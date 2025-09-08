# End-to-End Data Flow

## **Step 1: Dataset Registry and Preloading**

(Module: **DatasetLoading.py**)

* **Input:** Root dataset folder with subfolders containing:

  * `TileConfiguration.csv` (stage coordinates).
  * Tile images (grayscale/color).
* **Process:**

  1. `build_fixed_registry(root)` → builds a list of datasets (`DatasetEntry`).
  2. `preload_dataset(entry, …)` → for each dataset:

     * Loads coordinates (`TileConfiguration.csv`).
     * Detects tile neighbors.
     * Loads images (grayscale tiles).
     * Crops overlap regions for neighbors.
  3. `preload_all_for_cv(registry, …)` → preloads *all* datasets into memory as `PreloadedDataset` objects.
* **Output:** Dictionary: `{dataset_id: PreloadedDataset}`.

---

## **Step 2: Cross-Validation Setup**

(Module: **CrossValidationTesting.py**)

* **Input:** `registry` (datasets) and `preloaded_by_id` (preloaded data).
* **Process:**

  * `build_outer_folds(registry, preloaded_by_id)`:

    * Builds **outer LODO folds** (Leave-One-Dataset-Out).
    * One dataset = **test set**, rest = **train set**.
    * Computes domain distributions + candidate links.
  * `build_inner_folds(train_dataset_ids, registry)`:

    * Builds **inner LODO folds** inside the outer training pool.
    * Used for hyperparameter tuning.
* **Output:**

  * List of outer folds → each has train/test split.
  * For each outer fold → list of inner folds (train/valid split).

---

## **Step 3: Stitching Pipeline Run**

(Module: **DatasetLoading.py**)

* **Input:** `dataset_id`, `preloaded_by_id`, hyperparameters.
* **Process (`run_stitching_pipeline_dataset_aware`)**:

  1. **Tile matching:** Runs SIFT on cropped overlaps.
  2. **Filtering:** Applies ratio test, min matches.
  3. **Conversion:** Converts pixel shifts → microns.
  4. **Optimization:** Solves for corrected coordinates (via linear system with regularization).
* **Output:**

  * `x_corrected` (optimized coordinates).
  * `x_noisy` (original stage coordinates).
  * `(A, b)` linear system.
  * `filtered_results` (kept matches).

---

## **Step 4: Scoring a Dataset**

(Module: **ObjectiveCreation.py**)

* **Input:** `(x_corrected, x_noisy, A, b, filtered_results, neighbors, overlaps)`
* **Scoring Functions:**

  * **Constraint error:** How well corrected coords satisfy displacement rules.
  * **Displacement error:** Deviation of corrected from original coords.
  * **Penalty:** Missing matches between neighbors (cardinals weighted more).
  * **Diagnostics:** Overlap similarity (NCC), coverage, dropped links.
* **Output:**

  * Scalar score (single-objective) **or**
  * Tuple of 3 objectives: `(constraint RMS, normalized displacement RMS, drop rate)` (multi-objective).

---

## **Step 5: Optuna Optimization**

(Module: **ObjectiveCreation.py**)

* **Process:**

  1. **Objective trial:**

     * Optuna proposes hyperparameters (λ, weight method, ratio threshold, min matches, etc.).
     * Runs stitching pipeline on **inner folds**.
     * Averages validation scores.
  2. **Pruning:** Bad trials are cut early (MedianPruner); Pruning doesnt work for multi-objective hyperparameter optimization
  3. **Multi-objective option:** Uses NSGA-II to find Pareto-optimal trade-offs between objectives.
* **Output:**

  * Best hyperparameters found by Optuna.
  * Per-trial diagnostics (coverage, dropped links, similarity, etc.).

---

# Big Picture: Flow Summary

1. **Load datasets** → (`DatasetLoading`) builds registry & preloads images, coordinates, neighbors, overlaps.
2. **Build folds** → (`CrossValidationTesting`) defines outer/inner CV splits.
3. **Run stitching** → (`DatasetLoading.run_stitching_pipeline_dataset_aware`) runs SIFT + filtering + optimization.
4. **Score results** → (`ObjectiveCreation`) computes errors, penalties, similarity, coverage.
5. **Tune hyperparameters** → (`ObjectiveCreation.objective` / `multi_objective`) drives Optuna optimization loop.

---

In short:
**Raw tile images + stage coordinates → \[DatasetLoading] → CV splits → \[CrossValidationTesting] → Stitching run → \[DatasetLoading] → Error/penalty scoring → \[ObjectiveCreation] → Hyperparameter search (Optuna).**

