import numpy as np
from math import sqrt, log
import matplotlib.pyplot as plt

class Optimization:
    def __init__(self, tile_df, filtered_matches, neighbor_map):
        self.tile_df = tile_df
        self.filtered_matches = filtered_matches
        self.neighbor_map = neighbor_map

        self.weights = None
        self.noisy_points = None
        self.rules = None
        self.tile_df_filtered = None
        self.A = None
        self.b = None
        self.corrected_points = None
        self.residuals = None

    def compute_rule_weights(self, method="hybrid", log_cap=50, normalize=True):
        raw_weights = {}
        for key, result in self.filtered_matches.items():
            inliers = result["num_inliers"]
            matches = result["num_matches"]

            if method == "raw_inliers":
                w = inliers
            elif method == "sqrt_inliers":
                w = sqrt(inliers)
            elif method == "inlier_ratio":
                w = inliers / matches if matches > 0 else 0.0
            elif method == "hybrid":
                ratio = inliers / matches if matches > 0 else 0.0
                w = inliers * ratio
            elif method == "log_capped":
                w = log(1 + min(inliers, log_cap))
            elif method in ["none", "uniform"]:
                w = 1.0
            else:
                raise ValueError(f"Unknown method: {method}")

            raw_weights[key] = w

        if normalize and method not in ["none", "uniform"]:
            values = np.array(list(raw_weights.values()))
            max_val = values.max() if len(values) > 0 else 1.0
            raw_weights = {k: v / max_val for k, v in raw_weights.items()}

        self.weights = raw_weights

    def prepare_optimization_inputs(self):
        connected_tile_names = list(self.neighbor_map.keys())

        self.tile_df_filtered = self.tile_df[self.tile_df['tile_name'].isin(connected_tile_names)].copy()
        self.tile_df_filtered = (
            self.tile_df_filtered.set_index('tile_name')
            .loc[connected_tile_names]
            .reset_index()
        )

        self.noisy_points = self.tile_df_filtered[['x', 'y']].to_numpy(dtype=float)

        name_to_index = {
            name: idx for idx, name in enumerate(self.tile_df_filtered['tile_name'].tolist())
        }

        self.rules = []
        for (tile_i, tile_j), result in self.filtered_matches.items():
            if tile_i not in name_to_index or tile_j not in name_to_index:
                continue
            i = name_to_index[tile_i]
            j = name_to_index[tile_j]
            dx = result["dx_microns"]
            dy = result["dy_microns"]
            w = self.weights.get((tile_i, tile_j), 1.0)
            self.rules.append((i, j, dx, dy, w))

    def build_system(self, lambda_reg=5.0, fix_tile=True):
        num_points = len(self.noisy_points)
        num_vars = 2 * num_points
        num_rule_eqs = 2 * len(self.rules)
        num_reg_eqs = 2 * num_points
        num_anchor_eqs = 2 if fix_tile else 0

        total_eqs = num_rule_eqs + num_reg_eqs + num_anchor_eqs

        A = np.zeros((total_eqs, num_vars), dtype=np.float64)
        b = np.zeros(total_eqs, dtype=np.float64)

        row = 0
        for i, j, dx, dy, w in self.rules:
            A[row, 2*i] = w
            A[row, 2*j] = -w
            b[row] = w * dx
            row += 1

            A[row, 2*i + 1] = w
            A[row, 2*j + 1] = -w
            b[row] = w * dy
            row += 1

        for i in range(num_points):
            A[row, 2*i] = lambda_reg
            b[row] = lambda_reg * self.noisy_points[i, 0]
            row += 1

            A[row, 2*i + 1] = lambda_reg
            b[row] = lambda_reg * self.noisy_points[i, 1]
            row += 1

        if fix_tile:
            A[row, 0] = 1.0
            b[row] = self.noisy_points[0, 0]
            row += 1

            A[row, 1] = 1.0
            b[row] = self.noisy_points[0, 1]
            row += 1

        self.A = A
        self.b = b

    def solve(self, visualize=False, lambda_reg=None):
        try:
            solution, residuals, rank, s = np.linalg.lstsq(self.A, self.b, rcond=None)
            self.corrected_points = solution.reshape((-1, 2))
            self.residuals = residuals

            if visualize:
                fig, ax = plt.subplots(figsize=(10, 10))
                ax.scatter(self.noisy_points[:, 0], self.noisy_points[:, 1], c='blue', label='Original', s=80, alpha=0.5)
                ax.scatter(self.corrected_points[:, 0], self.corrected_points[:, 1], c='red', label='Corrected', s=80)
                for i in range(len(self.noisy_points)):
                    ax.plot(
                        [self.noisy_points[i, 0], self.corrected_points[i, 0]],
                        [self.noisy_points[i, 1], self.corrected_points[i, 1]],
                        linestyle='--', color='gray', alpha=0.6
                    )
                ax.set_title(f"Optimization (λ = {lambda_reg})", fontsize=14)
                ax.axis('equal')
                ax.legend()
                plt.gca().invert_yaxis()
                plt.grid(True)
                plt.show()

            return self.corrected_points, self.residuals

        except np.linalg.LinAlgError as e:
            print(f"[Error] Optimization failed: {e}")
            return None, None

    def run(self, lambda_reg=5.0, weight_method="hybrid", normalize=True, fix_tile=True, visualize=False):
        self.compute_rule_weights(method=weight_method, normalize=normalize)
        self.prepare_optimization_inputs()
        self.build_system(lambda_reg=lambda_reg, fix_tile=fix_tile)
        return self.solve(visualize=visualize, lambda_reg=lambda_reg)
