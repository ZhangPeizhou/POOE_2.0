#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, pickle, argparse
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    precision_score, recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, roc_auc_score, average_precision_score
)
import matplotlib.pyplot as plt
plt.switch_backend("Agg")

# ================= 路径与固定配置（grid 版，覆盖写同名文件） =================
BASE_DIR   = "/content/POOE_2.0/ESM2_All_Data_5-Folds"
RESULT_DIR = os.path.join(BASE_DIR, "results_cost_sensitive_grid")
MODEL_DIR  = os.path.join(BASE_DIR, "models")

MODEL_RAW_PATH  = os.path.join(MODEL_DIR, "svm_cs_grid_final.joblib")
MODEL_WITH_THR  = os.path.join(MODEL_DIR, "svm_cs_grid_final_with_threshold.joblib")
BEST_THR_JSON   = os.path.join(RESULT_DIR, "best_threshold_cs_grid.json")
GRID_SUMMARY    = os.path.join(RESULT_DIR, "grid_search_summary.json")

EXTERNAL = {
    "eff_pos": "/content/POOE_2.0/EffectorP-3.0-Data/TestData_Embedding_ESM2/positivedata_external_test.pkl",
    "eff_neg": "/content/POOE_2.0/EffectorP-3.0-Data/TestData_Embedding_ESM2/negativedata_external_test.pkl",
    "fun_pos": "/content/POOE_2.0/Fungtion-Data/Fungtion_Independent_Embedding_ESM2/positivedata_fungtion.pkl",
    "fun_neg": "/content/POOE_2.0/Fungtion-Data/Fungtion_Independent_Embedding_ESM2/negativedata_fungtion.pkl",
}

# ================= Cost-sensitive & 受限小网格 =================
CLASS_WEIGHT = "balanced"
# 只在“好点（C=10, gamma=0.25）”周围细化
GRID_C_VALUES = [7.5, 10, 15, 22]
GRID_GAMMA_VALUES = [0.15, 0.2, 0.25, 0.3]
GRID_SCORING = "average_precision"
GRID_VERBOSE = 2  # 打印进度
GRID_CV_FOLDS = 3

# 基线点（退化保护的回退参数）
BASELINE_PARAMS = {"C": 10.0, "gamma": 0.25}

# 阈值扫描（基于 decision_function）
PCT_LOW, PCT_HIGH = 1.0, 99.0
N_THR = 200
SELECT_METRIC = "mcc"  # 可改为 'f1'

# ================= 工具函数 =================
def ensure_dirs():
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_fold_data(fold_num: int) -> Tuple[Dict, Dict, Dict, Dict]:
    fold_path = os.path.join(BASE_DIR, f"fold{fold_num}_pkl")
    pos_tr = load_pickle(os.path.join(fold_path, f"positivedata_k{fold_num}.pkl"))
    pos_te = load_pickle(os.path.join(fold_path, f"positivedata_test_k{fold_num}.pkl"))
    neg_tr = load_pickle(os.path.join(fold_path, f"negativedata_k{fold_num}.pkl"))
    neg_te = load_pickle(os.path.join(fold_path, f"negativedata_test_k{fold_num}.pkl"))
    return pos_tr, pos_te, neg_tr, neg_te

def pad_or_truncate(features_list: List[np.ndarray], target_len: int) -> np.ndarray:
    fixed = []
    for feat in features_list:
        arr = np.array(feat)
        if arr.ndim > 1:
            arr = arr.flatten()
        if len(arr) > target_len:
            arr = arr[:target_len]
        elif len(arr) < target_len:
            arr = np.concatenate([arr, np.zeros(target_len - len(arr), dtype=arr.dtype)])
        fixed.append(arr)
    return np.asarray(fixed, dtype=np.float32)

def max_length_over_all_folds() -> int:
    max_len = 0
    for k in range(1, 6):
        pos_tr, pos_te, neg_tr, neg_te = load_fold_data(k)
        all_tr = list(pos_tr.values()) + list(neg_tr.values())
        all_te = list(pos_te.values()) + list(neg_te.values())
        local_max = max(max(map(len, all_tr)), max(map(len, all_te)))
        max_len = max(max_len, local_max)
    return max_len

def metrics_from_scores(y_true, scores, thr: float) -> Dict[str, float]:
    y_pred = (scores >= thr).astype(int)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)
    mcc       = matthews_corrcoef(y_true, y_pred)
    try:
        auroc = roc_auc_score(y_true, scores)
    except Exception:
        auroc = float("nan")
    try:
        auprc = average_precision_score(y_true, scores)
    except Exception:
        auprc = float("nan")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn)
    return {
        "threshold": thr,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "accuracy": acc,
        "f1": f1,
        "mcc": mcc,
        "auprc": auprc,
        "auroc": auroc
    }

def save_fold_summary_txt(df: pd.DataFrame, title: str, out_path: str):
    mean = df.mean(); std = df.std()
    lines = [
        "========================================",
        title,
        "Metric       |    Mean    |    Std    ",
        "----------------------------------------",
    ]
    for k in ["precision","recall","specificity","accuracy","mcc","f1","auroc","auprc"]:
        lines.append(f"{k:<12} | {mean[k]:.3f} ± {std[k]:.3f}")
    txt = "\n".join(lines)
    print(txt)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(txt)

def decide_n_jobs(user_n_jobs: Optional[int]) -> int:
    """自动并行策略：默认 2；若内存很富余且 CPU≥4，则给 4。"""
    if user_n_jobs is not None and user_n_jobs > 0:
        print(f"[INFO] Using user-specified n_jobs={user_n_jobs}")
        return user_n_jobs
    try:
        cpu_cnt = os.cpu_count() or 2
    except Exception:
        cpu_cnt = 2
    mem_gb = None
    try:
        import psutil
        mem_gb = psutil.virtual_memory().available / (1024**3)
    except Exception:
        pass
    n_jobs = 2 if cpu_cnt >= 2 else 1
    if mem_gb is not None:
        print(f"[INFO] Detected CPU={cpu_cnt}, available RAM≈{mem_gb:.1f} GB")
        if mem_gb >= 16 and cpu_cnt >= 4:
            n_jobs = min(4, cpu_cnt)
    else:
        print(f"[INFO] Detected CPU={cpu_cnt}, available RAM=Unknown")
    print(f"[INFO] Final n_jobs decided: {n_jobs}")
    return n_jobs

# ================= 主流程（受限小网格 + 退化保护） =================
def main():
    parser = argparse.ArgumentParser(description="Cost-Sensitive SVM (Constrained Grid with Regression Guard)")
    parser.add_argument("--n_jobs", type=int, default=None, help="并行作业数；不填则自动选择一个稳妥值")
    parser.add_argument("--cv", type=int, default=3, help="GridSearchCV 折数（默认3）")
    parser.add_argument("--select_metric", type=str, default=SELECT_METRIC, choices=["mcc", "f1"],
                        help="用哪个指标从 OOF 曲线上选阈值（默认 mcc）")
    args = parser.parse_args()

    ensure_dirs()

    # 1) 汇总训练数据
    final_input_len = max_length_over_all_folds()
    X_all, y_all_tr = [], []
    for k in range(1, 6):
        pos_tr, _, neg_tr, _ = load_fold_data(k)
        X_all.extend(list(pos_tr.values())); X_all.extend(list(neg_tr.values()))
        y_all_tr.extend([1]*len(pos_tr) + [0]*len(neg_tr))
    X_all = pad_or_truncate(X_all, final_input_len)
    y_all_tr = np.array(y_all_tr, dtype=int)

    n_jobs_use = decide_n_jobs(args.n_jobs)

    # 2) 先评估基线点的 CV AUPRC
    baseline_est = SVC(kernel="rbf", probability=False, class_weight=CLASS_WEIGHT,
                       C=BASELINE_PARAMS["C"], gamma=BASELINE_PARAMS["gamma"])
    baseline_scores = cross_val_score(baseline_est, X_all, y_all_tr,
                                      scoring=GRID_SCORING, cv=args.cv, n_jobs=n_jobs_use)
    baseline_score = float(np.mean(baseline_scores))
    print(f"[BASELINE] params={BASELINE_PARAMS}, cv={args.cv}, "
          f"{GRID_SCORING}={baseline_score:.6f}")

    # 3) 受限小网格搜索（进度+并行）
    params = {"C": GRID_C_VALUES, "gamma": GRID_GAMMA_VALUES}
    est = SVC(kernel="rbf", probability=False, class_weight=CLASS_WEIGHT)
    gs = GridSearchCV(
        est, params, scoring=GRID_SCORING, cv=args.cv,
        n_jobs=n_jobs_use, verbose=GRID_VERBOSE, refit=False
    )
    gs.fit(X_all, y_all_tr)
    best_idx = int(np.argmax(gs.cv_results_["mean_test_score"]))
    grid_best_params = {
        "C": float(gs.cv_results_["param_C"].data[best_idx]),
        "gamma": float(gs.cv_results_["param_gamma"].data[best_idx])
    }
    grid_best_score = float(gs.cv_results_["mean_test_score"][best_idx])

    # 4) 退化保护：若网格最优 < 基线 → 回退到基线参数
    if grid_best_score < baseline_score:
        C_use, gamma_use = BASELINE_PARAMS["C"], BASELINE_PARAMS["gamma"]
        chosen_from = "baseline_fallback"
        chosen_cv_score = baseline_score
        print(f"[GUARD] Grid best ({grid_best_score:.6f}) < baseline ({baseline_score:.6f}), "
              f"fallback to baseline params {BASELINE_PARAMS}")
    else:
        C_use, gamma_use = grid_best_params["C"], grid_best_params["gamma"]
        chosen_from = "grid_best"
        chosen_cv_score = grid_best_score
        print(f"[GRID] Using grid best params: {grid_best_params} ({GRID_SCORING}={grid_best_score:.6f})")

    # 保存搜索摘要（覆盖写）
    with open(GRID_SUMMARY, "w", encoding="utf-8") as f:
        json.dump({
            "grid": {"C": GRID_C_VALUES, "gamma": GRID_GAMMA_VALUES,
                     "cv_folds": args.cv, "scoring": GRID_SCORING},
            "baseline": {"params": BASELINE_PARAMS, "cv_score(auprc)": baseline_score},
            "grid_best": {"params": grid_best_params, "cv_score(auprc)": grid_best_score},
            "chosen": {"from": chosen_from, "params": {"C": C_use, "gamma": gamma_use},
                       "cv_score(auprc)": chosen_cv_score},
            "class_weight": CLASS_WEIGHT,
            "n_jobs": n_jobs_use
        }, f, ensure_ascii=False, indent=2)

    # 5) 用最终选定的 C,gamma 生成 OOF 分数（5 folds）
    y_all, s_all = [], []
    fold_rows_thr0, fold_rows_oof = [], []
    for k in range(1, 6):
        pos_tr, pos_te, neg_tr, neg_te = load_fold_data(k)
        X_tr_raw = list(pos_tr.values()) + list(neg_tr.values())
        y_tr     = np.array([1]*len(pos_tr) + [0]*len(neg_tr), dtype=int)
        X_te_raw = list(pos_te.values()) + list(neg_te.values())
        y_te     = np.array([1]*len(pos_te) + [0]*len(neg_te), dtype=int)
        X_tr = pad_or_truncate(X_tr_raw, final_input_len)
        X_te = pad_or_truncate(X_te_raw, final_input_len)

        clf = SVC(kernel="rbf", C=C_use, gamma=gamma_use,
                  probability=False, class_weight=CLASS_WEIGHT)
        clf.fit(X_tr, y_tr)
        s_te = clf.decision_function(X_te)

        y_all.append(y_te); s_all.append(s_te)
        fold_rows_thr0.append(metrics_from_scores(y_te, s_te, 0.0))

    y_all = np.concatenate(y_all); s_all = np.concatenate(s_all)
    print(f"[INFO] OOF size={len(y_all)}, pos_ratio={y_all.mean():.4f}")

    # 6) OOF 阈值扫描 → 选最优
    lo = np.percentile(s_all, PCT_LOW); hi = np.percentile(s_all, PCT_HIGH)
    thr_grid = np.linspace(lo, hi, N_THR)
    rows, best_val, best_row = [], -1e9, None
    for thr in thr_grid:
        mts = metrics_from_scores(y_all, s_all, thr); rows.append(mts)
        val = mts[args.select_metric]
        if val > best_val:
            best_val, best_row = val, dict(mts)

    df_grid = pd.DataFrame(rows)
    grid_csv = os.path.join(RESULT_DIR, "threshold_grid_cs_grid.csv")
    df_grid.to_csv(grid_csv, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {grid_csv}")

    for metric_name in ["f1", "mcc"]:
        plt.figure(figsize=(9, 6))
        plt.plot(df_grid["threshold"].values, df_grid[metric_name].values)
        plt.xlabel("Threshold (decision_function)")
        plt.ylabel(metric_name.upper())
        plt.title(f"{metric_name.upper()} vs Threshold (CS-SVM Grid, C={C_use}, gamma={gamma_use})")
        out_png = os.path.join(RESULT_DIR, f"curve_cs_grid_{metric_name}.png")
        plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()
        print(f"[SAVE] {out_png}")

    best_thr = float(best_row["threshold"])
    with open(BEST_THR_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "best_threshold": best_thr,
            "metric": args.select_metric,
            "svm_params": {"C": C_use, "gamma": gamma_use, "class_weight": CLASS_WEIGHT},
            "n_jobs": n_jobs_use, "cv": args.cv, "chosen_from": chosen_from
        }, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Best threshold by {args.select_metric} (OOF): {best_thr:.6f}")
    print("[INFO] OOF Metrics @ best threshold:")
    print({k: round(best_row[k], 4) for k in ["precision","recall","specificity","accuracy","f1","mcc","auprc","auroc"]})

    # 7) 回写每折（thr=0.0 与 OOF-fixed）的 5-Fold Summary
    for k in range(1, 6):
        pos_tr, pos_te, neg_tr, neg_te = load_fold_data(k)
        X_tr_raw = list(pos_tr.values()) + list(neg_tr.values())
        y_tr     = np.array([1]*len(pos_tr) + [0]*len(neg_tr), dtype=int)
        X_te_raw = list(pos_te.values()) + list(neg_te.values())
        y_te     = np.array([1]*len(pos_te) + [0]*len(neg_te), dtype=int)
        X_tr = pad_or_truncate(X_tr_raw, final_input_len)
        X_te = pad_or_truncate(X_te_raw, final_input_len)
        clf = SVC(kernel="rbf", C=C_use, gamma=gamma_use, probability=False, class_weight=CLASS_WEIGHT)
        clf.fit(X_tr, y_tr); s_te = clf.decision_function(X_te)
        fold_rows_oof.append(metrics_from_scores(y_te, s_te, best_thr))

    save_fold_summary_txt(pd.DataFrame(fold_rows_thr0),
                          "5-Fold Summary [Cost-Sensitive SVM (Grid)] (thr=0.0, decision_function)",
                          os.path.join(RESULT_DIR, "fold_summary_thr0.txt"))
    save_fold_summary_txt(pd.DataFrame(fold_rows_oof),
                          f"5-Fold Summary [Cost-Sensitive SVM (Grid)] (thr=OOF-fixed, {best_thr:.6f})",
                          os.path.join(RESULT_DIR, "fold_summary_oof.txt"))

    # 8) 全量训练最终模型并保存（覆盖写）
    import joblib
    final_clf = SVC(kernel="rbf", C=C_use, gamma=gamma_use, probability=False, class_weight=CLASS_WEIGHT)
    final_clf.fit(X_all, y_all_tr)
    joblib.dump(final_clf, MODEL_RAW_PATH)
    print(f"[SAVE] {MODEL_RAW_PATH}")

    joblib.dump({
        "model": final_clf,
        "threshold": best_thr,
        "score_type": "decision_function",
        "input_len": final_input_len,
        "svm_params": {"C": C_use, "gamma": gamma_use, "class_weight": CLASS_WEIGHT},
        "n_jobs": n_jobs_use, "cv": args.cv, "chosen_from": chosen_from
    }, MODEL_WITH_THR)
    print(f"[SAVE] {MODEL_WITH_THR}")

    # 9) External（若四个 pkl 都存在就评估；否则跳过）
    try:
        for p in EXTERNAL.values():
            if not os.path.exists(p):
                raise FileNotFoundError(p)

        def load_vec_list(p):
            obj = load_pickle(p)
            return list(obj.values()) if isinstance(obj, dict) else obj

        X_pos = load_vec_list(EXTERNAL["eff_pos"]) + load_vec_list(EXTERNAL["fun_pos"])
        X_neg = load_vec_list(EXTERNAL["eff_neg"]) + load_vec_list(EXTERNAL["fun_neg"])
        X_te_raw = X_pos + X_neg
        y_te     = np.array([1]*len(X_pos) + [0]*len(X_neg), dtype=int)
        X_te     = pad_or_truncate(X_te_raw, final_input_len)

        s_te = final_clf.decision_function(X_te)

        def eval_at_thr(scores, thr):
            y_pred = (scores >= thr).astype(int)
            mts = {
                "precision": precision_score(y_te, y_pred, zero_division=0),
                "recall": recall_score(y_te, y_pred, zero_division=0),
                "f1": f1_score(y_te, y_pred, zero_division=0),
                "mcc": matthews_corrcoef(y_te, y_pred),
                "auroc": roc_auc_score(y_te, scores),
                "auprc": average_precision_score(y_te, scores)
            }
            tn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()
            mts["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            mts["accuracy"] = (tp + tn) / (tp + tn + fp + fn)
            return mts

        mts_best = eval_at_thr(s_te, best_thr)
        mts_0    = eval_at_thr(s_te, 0.0)

        pd.DataFrame({
            "y_true": y_te,
            "score": s_te,
            "pred_thr_best": (s_te >= best_thr).astype(int),
            "pred_thr_0": (s_te >= 0.0).astype(int)
        }).to_csv(os.path.join(RESULT_DIR, "external_scores_preds_cs_grid.csv"), index=False, encoding="utf-8-sig")

        with open(os.path.join(RESULT_DIR, "external_summary_cs_grid.json"), "w", encoding="utf-8") as f:
            json.dump({
                "best_threshold": best_thr,
                "metrics_at_best": {k: float(v) for k, v in mts_best.items()},
                "metrics_at_0": {k: float(v) for k, v in mts_0.items()},
                "svm_params": {"C": C_use, "gamma": gamma_use, "class_weight": CLASS_WEIGHT},
                "n_jobs": n_jobs_use, "cv": args.cv, "chosen_from": chosen_from
            }, f, ensure_ascii=False, indent=2)

        print("\n[EXTERNAL] Metrics @ best threshold (OOF-chosen)")
        for k in ["precision","recall","specificity","accuracy","f1","mcc","auroc","auprc"]:
            print(f"  {k:>11}: {mts_best[k]:.4f}")
        print("[EXTERNAL] Metrics @ 0.0 threshold (default)")
        for k in ["precision","recall","specificity","accuracy","f1","mcc","auroc","auprc"]:
            print(f"  {k:>11}: {mts_0[k]:.4f}")

    except FileNotFoundError as e:
        print(f"[INFO] External test skipped (missing file: {e})")

    print("\n[DONE] CS-SVM（Constrained Grid + Regression Guard）→ OOF 选阈值 → 5-Fold 汇总 → 最终模型 → External 评估 完成。")

if __name__ == "__main__":
    main()
