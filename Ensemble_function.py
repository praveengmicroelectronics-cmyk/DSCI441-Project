import os
import glob
import numpy as np
import pandas as pd

_DEFAULT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROB_SUM_TOL = 1e-4


def validate_csv_files(
    search_dir: str = _DEFAULT_DIR,
    pattern: str = "*_probabilities.csv",
    prob_sum_tol: float = _PROB_SUM_TOL,
) -> dict:
    csv_paths = sorted(glob.glob(os.path.join(search_dir, pattern)))
    if not csv_paths:
        return {
            "files": [], "model_names": [], "per_file": {}, "cross_file": {},
            "passed": False,
            "summary": f"ERROR: No files matching '{pattern}' in {search_dir}",
        }

    def _label(path):
        stem = os.path.splitext(os.path.basename(path))[0]
        return stem[: -len("_probabilities")] if stem.endswith("_probabilities") else stem

    model_names = [_label(p) for p in csv_paths]
    all_ok      = True
    per_file    = {}
    loaded_dfs  = {}

    for path, model_name in zip(csv_paths, model_names):
        df     = pd.read_csv(path)
        checks = {}

        for col in ("sample_idx", "true_label", "pred_label"):
            checks[f"has_{col}"] = col in df.columns

        prob_cols    = [c for c in df.columns if c.startswith("prob_")]
        this_classes = [c[len("prob_"):] for c in prob_cols]
        checks["has_prob_cols"] = len(prob_cols) > 0
        checks["n_prob_cols"]   = len(prob_cols)
        checks["class_names"]   = this_classes

        if not all(checks[f"has_{c}"] for c in ("sample_idx", "true_label", "pred_label")) \
                or not checks["has_prob_cols"]:
            checks["skipped_further"] = True
            per_file[model_name] = checks
            all_ok = False
            continue

        df_sorted = df.sort_values("sample_idx").reset_index(drop=True)
        loaded_dfs[model_name] = df_sorted

        n_dup = int(df_sorted["sample_idx"].duplicated().sum())
        checks["duplicate_sample_idx"] = n_dup
        checks["no_duplicates_ok"]     = n_dup == 0

        prob_vals = df_sorted[prob_cols].values.astype(float)
        n_nan = int(np.isnan(prob_vals).sum())
        n_inf = int(np.isinf(prob_vals).sum())
        checks["n_nan_in_probs"] = n_nan
        checks["n_inf_in_probs"] = n_inf
        checks["no_nan_ok"]      = n_nan == 0
        checks["no_inf_ok"]      = n_inf == 0

        n_neg = int((prob_vals < 0).sum())
        checks["n_negative_probs"] = n_neg
        checks["no_negatives_ok"]  = n_neg == 0

        row_sums   = prob_vals.sum(axis=1)
        bad_sums   = np.abs(row_sums - 1.0) > prob_sum_tol
        n_bad_sums = int(bad_sums.sum())
        checks["n_rows_bad_prob_sum"] = n_bad_sums
        checks["prob_sum_ok"]         = n_bad_sums == 0
        if n_bad_sums > 0:
            checks["prob_sum_range"] = (
                round(float(row_sums.min()), 6),
                round(float(row_sums.max()), 6),
            )

        unique_labels  = set(df_sorted["true_label"].astype(str).unique())
        unknown_labels = unique_labels - set(this_classes)
        checks["unknown_labels"] = sorted(unknown_labels)
        checks["true_labels_ok"] = len(unknown_labels) == 0

        checks["n_rows"]           = len(df_sorted)
        checks["sample_idx_range"] = (
            int(df_sorted["sample_idx"].min()),
            int(df_sorted["sample_idx"].max()),
        )

        file_ok = all([
            checks["no_duplicates_ok"],
            checks["no_nan_ok"],
            checks["no_inf_ok"],
            checks["no_negatives_ok"],
            checks["prob_sum_ok"],
            checks["true_labels_ok"],
        ])
        checks["file_passed"] = file_ok
        if not file_ok:
            all_ok = False

        per_file[model_name] = checks

    cross        = {}
    valid_models = [m for m in model_names if m in loaded_dfs]

    if len(valid_models) < 2:
        cross["skipped"] = "Fewer than 2 valid files — cross-file checks skipped."
    else:
        ref_name    = valid_models[0]
        ref_df      = loaded_dfs[ref_name]
        ref_idx     = ref_df["sample_idx"].values
        ref_y       = ref_df["true_label"].astype(str).values
        ref_classes = per_file[ref_name]["class_names"]

        row_counts = {m: len(loaded_dfs[m]) for m in valid_models}
        cross["row_counts"]    = row_counts
        cross["row_counts_ok"] = len(set(row_counts.values())) == 1

        class_mismatches = {}
        for m in valid_models[1:]:
            tc = per_file[m]["class_names"]
            if tc != ref_classes:
                class_mismatches[m] = {"expected": ref_classes, "got": tc}
        cross["class_order_ref"]        = ref_classes
        cross["class_order_mismatches"] = class_mismatches
        cross["class_order_ok"]         = len(class_mismatches) == 0

        idx_set_mismatches = {}
        for m in valid_models[1:]:
            their_idx   = set(loaded_dfs[m]["sample_idx"].values.tolist())
            ref_idx_set = set(ref_idx.tolist())
            only_ref    = ref_idx_set - their_idx
            only_theirs = their_idx - ref_idx_set
            if only_ref or only_theirs:
                idx_set_mismatches[m] = {
                    "only_in_ref": len(only_ref), "only_in_theirs": len(only_theirs)
                }
        cross["sample_idx_set_mismatches"] = idx_set_mismatches
        cross["sample_idx_set_ok"]         = len(idx_set_mismatches) == 0

        idx_order_mismatches = {}
        if cross["row_counts_ok"] and cross["sample_idx_set_ok"]:
            for m in valid_models[1:]:
                their_idx = loaded_dfs[m]["sample_idx"].values
                if not np.array_equal(ref_idx, their_idx):
                    idx_order_mismatches[m] = int(np.sum(ref_idx != their_idx))
        cross["sample_idx_order_mismatches"] = idx_order_mismatches
        cross["sample_idx_order_ok"]         = len(idx_order_mismatches) == 0

        label_mismatches = {}
        if cross["row_counts_ok"] and cross["sample_idx_order_ok"]:
            for m in valid_models[1:]:
                their_y = loaded_dfs[m]["true_label"].astype(str).values
                n_diff  = int(np.sum(ref_y != their_y))
                if n_diff > 0:
                    label_mismatches[m] = n_diff
        cross["true_label_mismatches"] = label_mismatches
        cross["true_label_ok"]         = len(label_mismatches) == 0

        cross_ok = all([
            cross["row_counts_ok"],
            cross["class_order_ok"],
            cross["sample_idx_set_ok"],
            cross["sample_idx_order_ok"],
            cross["true_label_ok"],
        ])
        cross["cross_passed"] = cross_ok
        if not cross_ok:
            all_ok = False

    n_pass  = sum(1 for v in per_file.values() if v.get("file_passed", False))
    summary = (
        f"{'ALL CHECKS PASSED' if all_ok else 'ISSUES FOUND'}  |  "
        f"{n_pass}/{len(csv_paths)} files passed per-file checks  |  "
        f"{len(csv_paths)} files scanned"
    )
    return {
        "files": csv_paths, "model_names": model_names,
        "per_file": per_file, "cross_file": cross,
        "passed": all_ok, "summary": summary,
    }


def load_meta_features(
    search_dir: str = _DEFAULT_DIR,
    pattern: str = "*_probabilities.csv",
    run_validation: bool = True,
) -> tuple:
    if run_validation:
        report = validate_csv_files(search_dir=search_dir, pattern=pattern)
        if not report["passed"]:
            raise ValueError(
                f"Consistency checks failed before loading.\n{report['summary']}\n"
                "Call validate_csv_files() for the full report."
            )

    csv_paths = sorted(glob.glob(os.path.join(search_dir, pattern)))
    if not csv_paths:
        raise FileNotFoundError(f"No files matching '{pattern}' found in: {search_dir}")

    def _label(path):
        stem = os.path.splitext(os.path.basename(path))[0]
        return stem[: -len("_probabilities")] if stem.endswith("_probabilities") else stem

    prob_blocks = []
    model_names = []
    ref_y       = None
    class_names = None

    for path in csv_paths:
        model_name = _label(path)
        df         = pd.read_csv(path).sort_values("sample_idx").reset_index(drop=True)
        prob_cols  = [c for c in df.columns if c.startswith("prob_")]

        if class_names is None:
            class_names = [c[len("prob_"):] for c in prob_cols]
            ref_y       = df["true_label"].astype(str).values

        prob_blocks.append(df[prob_cols].values.astype(float))
        model_names.append(model_name)
        print(f"  Loaded {model_name}: {len(df)} rows x {len(prob_cols)} prob cols", flush=True)

    meta_X = np.hstack(prob_blocks)
    print(
        f"\nmeta_X shape : {meta_X.shape}  ({len(model_names)} models x {len(class_names)} classes)\n"
        f"y shape      : {ref_y.shape}\nModels       : {model_names}\nClasses      : {class_names}",
        flush=True,
    )
    return meta_X, ref_y, model_names, class_names


def meta_feature_dataframe(
    search_dir: str = _DEFAULT_DIR,
    pattern: str = "*_probabilities.csv",
) -> tuple:
    meta_X, y, model_names, class_names = load_meta_features(
        search_dir=search_dir, pattern=pattern
    )
    col_names = [f"{model}__prob_{c}" for model in model_names for c in class_names]
    return pd.DataFrame(meta_X, columns=col_names), y


if __name__ == "__main__":
    SEP = "=" * 70
    print(SEP)
    print("Ensemble_function.py — full consistency validation")
    print(SEP)

    report = validate_csv_files()
    print(f"\n{report['summary']}\n")

    print("-- Per-file checks " + "-" * 51)
    for model, chk in report["per_file"].items():
        status = "PASS" if chk.get("file_passed") else "FAIL"
        print(f"\n  [{status}] {model}")
        print(f"         rows           : {chk.get('n_rows', '?')}")
        print(f"         sample_idx range: {chk.get('sample_idx_range', '?')}")
        print(f"         classes        : {chk.get('class_names', '?')}")
        print(f"         n_prob_cols    : {chk.get('n_prob_cols', '?')}")
        print(f"         duplicates     : {chk.get('duplicate_sample_idx', '?')}")
        print(f"         NaN in probs   : {chk.get('n_nan_in_probs', '?')}")
        print(f"         Inf in probs   : {chk.get('n_inf_in_probs', '?')}")
        print(f"         negative probs : {chk.get('n_negative_probs', '?')}")
        print(f"         bad row sums   : {chk.get('n_rows_bad_prob_sum', '?')}")
        if chk.get("n_rows_bad_prob_sum", 0) > 0:
            print(f"         sum range      : {chk.get('prob_sum_range', '?')}")
        if chk.get("unknown_labels"):
            print(f"         unknown labels : {chk['unknown_labels']}")

    cross = report["cross_file"]
    print(f"\n-- Cross-file checks " + "-" * 49)
    if "skipped" in cross:
        print(f"  {cross['skipped']}")
    else:
        status = "PASS" if cross.get("cross_passed") else "FAIL"
        print(f"  [{status}]")
        print(f"  Row counts               : {cross['row_counts']}")
        print(f"  Row counts consistent    : {cross['row_counts_ok']}")
        print(f"  Class order ref          : {cross['class_order_ref']}")
        print(f"  Class order OK           : {cross['class_order_ok']}")
        if cross["class_order_mismatches"]:
            print(f"  Class order mismatches   : {cross['class_order_mismatches']}")
        print(f"  sample_idx set OK        : {cross['sample_idx_set_ok']}")
        if cross["sample_idx_set_mismatches"]:
            print(f"  sample_idx set issues    : {cross['sample_idx_set_mismatches']}")
        print(f"  sample_idx order OK      : {cross['sample_idx_order_ok']}")
        if cross["sample_idx_order_mismatches"]:
            print(f"  sample_idx order issues  : {cross['sample_idx_order_mismatches']}")
        print(f"  true_label consistent    : {cross['true_label_ok']}")
        if cross["true_label_mismatches"]:
            print(f"  true_label mismatches    : {cross['true_label_mismatches']}")

    if not report["passed"]:
        print(f"\nVALIDATION FAILED")
        print("Fix the issues above before using load_meta_features().")
    else:
        print(f"\nAll checks passed — loading meta-feature matrix...\n")
        meta_df, y = meta_feature_dataframe()
        print(f"\nmeta_df shape  : {meta_df.shape}")
        print(f"y shape        : {y.shape}")
        print(f"\nClass distribution in y:")
        unique, counts = np.unique(y, return_counts=True)
        for cls, cnt in zip(unique, counts):
            print(f"  {cls:20s} : {cnt}")
        n_nan = int(np.isnan(meta_df.values).sum())
        n_inf = int(np.isinf(meta_df.values).sum())
        print(f"\nNaN in meta_X  : {n_nan}")
        print(f"Inf in meta_X  : {n_inf}")
        print("\nDone. meta_X and y are ready for a meta-learner.")
