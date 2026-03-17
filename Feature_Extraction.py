import numpy as np
from Grid_Generation import get_grid
from scipy.spatial.distance import pdist
def extract_wafer_features(wafer_map, params):

    wm = np.asarray(wafer_map)

    h, w = wm.shape

    x_norm, y_norm, r = get_grid((h, w))

    inside = wm != 0
    defect = wm == 2

    n_inside = float(np.sum(inside))
    n_defect = float(np.sum(defect))

    defect_density = n_defect / (n_inside + params["eps"])

    if n_defect == 0:
        return {
            "defect_density": 0.0,
            "centroid_x": 0.0,
            "centroid_y": 0.0,
            "radial_mean": 0.0,
            "radial_std": 0.0,
            "radial_skew": 0.0,
            "edge_ratio": 0.0,
            "q1": 0.0,
            "q2": 0.0,
            "q3": 0.0,
            "q4": 0.0,
            "spread": 0.0,
            "ring_score": 0.0,
            "center_ratio": 0.0,
            "angular_variance": 0.0,
            "cluster_density": 0.0
        }

    r_max = np.max(r[inside])

    r_def = r[defect] / (r_max + params["eps"])

    x_def = x_norm[defect]
    y_def = y_norm[defect]

    centroid_x = np.mean(x_def)
    centroid_y = np.mean(y_def)

    radial_mean = np.mean(r_def)
    radial_std = np.std(r_def)

    if radial_std < params["eps"]:
        radial_skew = 0.0
    else:
        z = (r_def - radial_mean) / (radial_std + params["eps"])
        radial_skew = np.mean(z**3)

    edge_ratio = np.mean(r_def >= params["edge_threshold"])

    q1 = np.mean((x_def >= 0) & (y_def >= 0))
    q2 = np.mean((x_def < 0) & (y_def >= 0))
    q3 = np.mean((x_def < 0) & (y_def < 0))
    q4 = np.mean((x_def >= 0) & (y_def < 0))

    # ---------------------------
    # Additional spatial features
    # ---------------------------

    dx = x_def - centroid_x
    dy = y_def - centroid_y

    spread = np.mean(np.sqrt(dx**2 + dy**2))

    ring_score = np.mean((r_def > 0.6) & (r_def < 0.9))

    center_ratio = np.mean(r_def < 0.3)

    angles = np.arctan2(y_def, x_def)

    angular_variance = np.var(angles)

    if len(x_def) > 1:
        cluster_density = 1 / (np.mean(pdist(np.column_stack([x_def, y_def]))) + params["eps"])
    else:
        cluster_density = 0

    return {
        "defect_density": defect_density,
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "radial_mean": radial_mean,
        "radial_std": radial_std,
        "radial_skew": radial_skew,
        "edge_ratio": edge_ratio,
        "q1": q1,
        "q2": q2,
        "q3": q3,
        "q4": q4,
        "spread": spread,
        "ring_score": ring_score,
        "center_ratio": center_ratio,
        "angular_variance": angular_variance,
        "cluster_density": cluster_density
    }