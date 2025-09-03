from typing import Dict, Tuple, Any
import numpy as np
import cv2

def match_cropped_tile_pairs(
    cropped_patches: Dict[str, Dict[str, np.ndarray]],
    neighbors: Dict[str, Dict[str, str]],
    ratio_thresh: float = 0.75,
    min_matches: int = 3
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher()

    opposite = {
        "left": "right", "right": "left",
        "up": "down", "down": "up",
        "top_left": "bottom_right", "top_right": "bottom_left",
        "bottom_left": "top_right", "bottom_right": "top_left"
    }

    results = {}

    for tile_name, directions in cropped_patches.items():
        for direction, patch in directions.items():
            neighbor_name = neighbors.get(tile_name, {}).get(direction)
            if neighbor_name is None:
                continue

            opp_dir = opposite.get(direction)
            if opp_dir is None:
                continue

            neighbor_patch = cropped_patches.get(neighbor_name, {}).get(opp_dir)
            if neighbor_patch is None:
                print(f"Missing patch: {neighbor_name} [{opp_dir}]")
                continue

            kp1, des1 = sift.detectAndCompute(patch, None)
            kp2, des2 = sift.detectAndCompute(neighbor_patch, None)

            if des1 is None or des2 is None:
                print(f"No descriptors for {tile_name} or {neighbor_name}")
                continue

            matches = bf.knnMatch(des1, des2, k=2)
            good_matches = [m for m, n in matches if m.distance < ratio_thresh * n.distance]

            match_key = (tile_name, neighbor_name)
            results[match_key] = {
                "direction": direction,
                "good_matches": good_matches,
                "num_matches": len(good_matches),
                "keypoints1": kp1,
                "keypoints2": kp2,
                "patch1": patch,
                "patch2": neighbor_patch,
                "affine_matrix": None,
                "dx": None,
                "dy": None,
                "rotation_deg": None,
                "num_inliers": 0
            }

            if len(good_matches) >= min_matches:
                pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                M, inliers = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC)
                if M is not None and inliers is not None:
                    dx, dy = M[0, 2], M[1, 2]
                    rotation_rad = np.arctan2(M[1, 0], M[0, 0])
                    rotation_deg = np.degrees(rotation_rad)

                    results[match_key].update({
                        "affine_matrix": M,
                        "dx": dx,
                        "dy": dy,
                        "rotation_deg": rotation_deg,
                        "num_inliers": int(np.sum(inliers))
                    })

    return results
