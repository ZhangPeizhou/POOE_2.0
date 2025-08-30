#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, pickle, argparse
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    precision_score, recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, roc_auc_score, average_precision_score
)
import matplotlib.pyplot as plt
plt.switch_backend("Agg")

# ======================== 顶部常量（与旧脚本保持一致） ========================
BASE_DIR    = "/content/POOE_2.0/ESM2_All_Data_5-Folds"
RESULT_DIR  = os.path.join(BASE_DIR, "results_cost_sensitive")   # 新：CS 结果单独目录
MODEL_DIR   = os.path.join(BASE_DIR, "models")

# 模型保存路径（不覆盖你原来的 svm_final.*）
MODEL_RAW_PATH   = os.path.join(MODEL_DIR, "svm_cs_final.joblib")
MODEL_WITH_THR   = os.path.join(MODEL_DIR, "svm_cs_final_with_threshold.joblib")
BEST_THR_JSON    = os.path.join(RESULT_DIR, "best_threshold_cs.json")

# 外测四个 pkl（与你之前一致）
EXTERNAL = {
    "eff_pos": "/content/POOE_2.0/EffectorP-3.0-Data/TestData_Embedding_ESM2/positivedata_external_test.pkl",
    "eff_neg": "/content/POOE_2.0/EffectorP-3.0-Data/TestData_Embedding_ESM2/negativedata_external_test.pkl",
    "fun_pos": "/content/POOE_2.0/Fungtion-Data/Fungtion_Independent_Embedding_ESM2/positivedata_fungtion.pkl",
    "fun_neg": "/content/POOE_2.0/Fungtion-Data/Fungtion_Independent_Embedding_ESM2/negativedata_fungtion.pkl",
}

# 原始参数 + cost-sensitive 关键：class_weight
SVM_C = 10.0
SVM_GAMMA = 0.25
CLASS_WEIGHT = "balanced"      # 也可改成 {0:1, 1:5} 等

# 小网格（可选，--grid 开启；便于复现）
GRID_C_VALUES = [3.0, 10.0, 30.0]
GRID_GAMMA_VALUES = [0.1, 0.25, 0.5]
GRID_CV_FOLDS = 3
GRID_SCORING = "average_precision"

# 阈值扫描（仍旧在 decision_function 分位区间扫描）
PCT_LOW, PCT_HIGH = 1.0, 99.0
N_THR = 200
SELECT_METRIC = "mcc"   # 用哪个指标挑“最佳阈值”：'mcc' 或 'f1'

# ======================== 工具函数 ========================
def ensure_dirs():
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

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

def load_fold_data(fold_num: int) -> Tuple[Dict, Dict, Dict, Dict]:
    fold_path = os.path.join(BASE_DIR, f"fold{fold_num}_pkl")
    pos_tr = load_pickle(os.path.join(fold_path, f"positivedata_k{fold_num}.pkl"))
    pos_te = load_pickle(os.path.join(fold_path, f"positivedata_test_k{fold_num}.pkl"))
    neg_tr = load_pickle(os.path.join(fold_path, f"negativedata_k{fold_num}.pkl"))
    neg_te = load_pickle(os.path.join(fold_path, f"negativedata_test_k{fold_num}.pkl"))
    return pos_tr, pos_te, neg_tr, neg_te

def max_length_over_all_folds() -> int:
    max_len = 0
    for k in range(1, 6):
        pos_tr, pos_te, neg_tr, neg_te = load_fold_data(k)
        all_tr = list(pos_tr.values()) + list(neg_tr.values())
        all_te = list(pos_te.values()) + list(neg_te.values())
        local_max = max(max(len(x) for x in all_tr), max(len(x) for x in all_te))
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

# ======================== 主流程 ========================
def main():
    parser = argparse.ArgumentParser(description="Cost-Sensitive SVM with decision_function thresholding")
    parser.add_argument("--grid", action="store_true", help="开启小网格 (C,gamma) 搜索（目标 AUPRC）")
    args = parser.parse_args()

    ensure_dirs()

    # 1) OOF 分数（5 折；与旧脚本一致），但模型加入 class_weight
    final_input_len = max_length_over_all_folds()
    y_all, s_all = [], []

    for k in range(1, 6):
        pos_tr, pos_te, neg_tr, neg_te = load_fold_data(k)

        X_tr_raw = list(pos_tr.values()) + list(neg_tr.values())
        y_tr     = np.array([1]*len(pos_tr) + [0]*len(neg_tr), dtype=int)

        X_te_raw = list(pos_te.values()) + list(neg_te.values())
        y_te     = np.array([1]*len(pos_te) + [0]*len(neg_te), dtype=int)

        X_tr = pad_or_truncate(X_tr_raw, final_input_len)
        X_te = pad_or_truncate(X_te_raw, final_input_len)

        clf = SVC(kernel="rbf",
                  C=SVM_C,
                  gamma=SVM_GAMMA,
                  probability=False,
                  class_weight=CLASS_WEIGHT)   # 关键
        clf.fit(X_tr, y_tr)

        s_te = clf.decision_function(X_te)  # 仍旧用 margin 分数

        y_all.append(y_te)
        s_all.append(s_te)

    y_all = np.concatenate(y_all)
    s_all = np.concatenate(s_all)
    print(f"[INFO] OOF size={len(y_all)}, pos_ratio={y_all.mean():.4f}")

    # 2) 在 OOF 分数分位区间扫描阈值（不泄漏 external）
    lo = np.percentile(s_all, PCT_LOW)
    hi = np.percentile(s_all, PCT_HIGH)
    thr_grid = np.linspace(lo, hi, N_THR)

    rows = []
    best_val = -1e9
    best_row = None
    for thr in thr_grid:
        mts = metrics_from_scores(y_all, s_all, thr)
        rows.append(mts)
        val = mts[SELECT_METRIC]
        if val > best_val:
            best_val = val
            best_row = dict(mts)

    df_grid = pd.DataFrame(rows)
    grid_csv = os.path.join(RESULT_DIR, "threshold_grid_cs.csv")
    df_grid.to_csv(grid_csv, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {grid_csv}")

    # 3) 画曲线（F1 与 MCC）
    for metric_name in ["f1", "mcc"]:
        plt.figure(figsize=(9, 6))
        plt.plot(df_grid["threshold"].values, df_grid[metric_name].values)
        plt.xlabel("Threshold (decision_function)")
        plt.ylabel(metric_name.upper())
        plt.title(f"{metric_name.upper()} vs Threshold (CS-SVM RBF, C={SVM_C}, gamma={SVM_GAMMA}, class_weight={CLASS_WEIGHT})")
        out_png = os.path.join(RESULT_DIR, f"curve_cs_{metric_name}.png")
        plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()
        print(f"[SAVE] {out_png}")

    best_thr = float(best_row["threshold"])
    print(f"[INFO] Best threshold by {SELECT_METRIC} (OOF): {best_thr:.6f}")
    print("[INFO] OOF Metrics @ best threshold:")
    print({k: round(best_row[k], 4) for k in ["precision","recall","specificity","accuracy","f1","mcc","auprc","auroc"]})

    # 4) （可选）小网格：在“全部训练数据上用 CV”挑更好的 C/gamma
    #    ——稳定、好解释；默认不开启
    import joblib
    # 先汇总全训练数据
    X_all, y_all_tr = [], []
    for k in range(1, 6):
        pos_tr, _, neg_tr, _ = load_fold_data(k)
        X_all.extend(list(pos_tr.values()))
        X_all.extend(list(neg_tr.values()))
        y_all_tr.extend([1]*len(pos_tr) + [0]*len(neg_tr))
    X_all = pad_or_truncate(X_all, final_input_len)
    y_all_tr = np.array(y_all_tr, dtype=int)

    C_use, gamma_use = SVM_C, SVM_GAMMA
    if args.grid:
        params = {"C": GRID_C_VALUES, "gamma": GRID_GAMMA_VALUES}
        est = SVC(kernel="rbf", probability=False, class_weight=CLASS_WEIGHT)
        gs = GridSearchCV(est, params, scoring=GRID_SCORING, cv=GRID_CV_FOLDS, n_jobs=-1, verbose=1)
        gs.fit(X_all, y_all_tr)
        C_use = float(gs.best_params_["C"])
        gamma_use = float(gs.best_params_["gamma"])
        with open(os.path.join(RESULT_DIR, "grid_search_summary.json"), "w", encoding="utf-8") as f:
            json.dump({
                "grid": {"C": GRID_C_VALUES, "gamma": GRID_GAMMA_VALUES, "cv_folds": GRID_CV_FOLDS, "scoring": GRID_SCORING},
                "best_params": gs.best_params_,
                "best_cv_score(auprc)": float(gs.best_score_)
            }, f, ensure_ascii=False, indent=2)
        print(f"[GRID] best: C={C_use}, gamma={gamma_use}")

    # 5) 用（可能更新后的）参数在“全部训练数据”上重训最终模型并保存
    final_clf = SVC(kernel="rbf", C=C_use, gamma=gamma_use, probability=False, class_weight=CLASS_WEIGHT)
    final_clf.fit(X_all, y_all_tr)

    joblib.dump(final_clf, MODEL_RAW_PATH)
    print(f"[SAVE] {MODEL_RAW_PATH}")

    joblib.dump({
        "model": final_clf,
        "threshold": best_thr,                 # OOF 选定阈值
        "score_type": "decision_function",
        "input_len": final_input_len,
        "svm_params": {"C": C_use, "gamma": gamma_use, "class_weight": CLASS_WEIGHT}
    }, MODEL_WITH_THR)
    print(f"[SAVE] {MODEL_WITH_THR}")

    with open(BEST_THR_JSON, "w", encoding="utf-8") as f:
        json.dump({"best_threshold": best_thr, "metric": SELECT_METRIC}, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] {BEST_THR_JSON}")

    # 6) External Test（若四个 pkl 都在就评估；否则跳过）
    try:
        for p in EXTERNAL.values():
            if not os.path.exists(p):
                raise FileNotFoundError(p)

        def load_vec_list(p):
            obj = load_pickle(p)
            if isinstance(obj, dict):
                return list(obj.values())
            return obj

        X_pos = load_vec_list(EXTERNAL["eff_pos"]) + load_vec_list(EXTERNAL["fun_pos"])
        X_neg = load_vec_list(EXTERNAL["eff_neg"]) + load_vec_list(EXTERNAL["fun_neg"])
        print(f"[INFO] External pos={len(X_pos)}, neg={len(X_neg)}, ratio={len(X_pos)/(len(X_pos)+len(X_neg)):.4f}")

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
        }).to_csv(os.path.join(RESULT_DIR, "external_scores_preds_cs.csv"), index=False, encoding="utf-8-sig")

        with open(os.path.join(RESULT_DIR, "external_summary_cs.json"), "w", encoding="utf-8") as f:
            json.dump({
                "best_threshold": best_thr,
                "metrics_at_best": {k: float(v) for k, v in mts_best.items()},
                "metrics_at_0": {k: float(v) for k, v in mts_0.items()},
                "svm_params": {"C": C_use, "gamma": gamma_use, "class_weight": CLASS_WEIGHT}
            }, f, ensure_ascii=False, indent=2)

        print("\n[EXTERNAL] Metrics @ best threshold (OOF-chosen)")
        for k in ["precision","recall","specificity","accuracy","f1","mcc","auroc","auprc"]:
            print(f"  {k:>11}: {mts_best[k]:.4f}")
        print("[EXTERNAL] Metrics @ 0.0 threshold (default)")
        for k in ["precision","recall","specificity","accuracy","f1","mcc","auroc","auprc"]:
            print(f"  {k:>11}: {mts_0[k]:.4f}")

    except FileNotFoundError as e:
        print(f"[INFO] External test skipped (missing file: {e})")

    print("\n[DONE] Cost-sensitive SVM → OOF 选阈值 → 保存新模型（含阈值） → External 评估（若有）完成。")


if __name__ == "__main__":
    main()
