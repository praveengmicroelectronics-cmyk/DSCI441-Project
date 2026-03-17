def get_feature_params():
    params={
    "edge_threshold": 0.80,   # radius from the center of the wafer to verify if a defect is near wafer edge
    "eps": 1e-12             # small constant used to avoid division-by-zero errors
    }
    return params
