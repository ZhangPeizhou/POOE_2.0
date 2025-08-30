#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict

from sklearn.svm import SVC
from sklearn.metrics import (
    precision_score, recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, roc_auc_score, average_precision_score
)

# ======================== 顶部常量（按需改） ========================
# 与现有工程结构一致（fold{n}_pkl/positivedata_k{n}.pkl 等）
BASE_DIR   = "/content/POOE_2.0/ESM2_All_Data_5-Folds"
RESULT_DIR = os.path.join(BASE_DIR, "results_threshold")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "svm_final.joblib")  # 与原路径一致，便于兼容你已有 external 流程

# 外测四个 pkl（你上条消息给的路径）
EXTERNAL = {
    "eff_pos": "/content/POOE_2.0/EffectorP-3.0-Data/TestData_Embedding_ESM2/positivedata_external_test.pkl",
    "eff_neg": "/content/POOE_2.0/EffectorP-3.0-Data/TestData_Embedding_ESM2/negativedata_external_test.pkl",
    "fun_pos": "/content/POOE_2.0/Fungtion-Data/Fungtion_Independent_Embedding_ESM2/positivedata_fungtion.pkl",
    "fun_neg": "/content/POOE_2.0/Fungtion-Data/Fungtion_Independent_Embedding_ESM2/negativedata_fungtion.pkl",
}

# SVM 固定参数（与之前一致）
SVM_C     = 10.0
SVM_GAMMA = 0.25

# 扫描的正类权重（负类权重固定为 1.0）。可按需删减或增加。
W_POS_LIST = [1, 2, 4, 8, 16]

# 阈值网格
THR_START, THR_END, THR_STEP = 0.05, 0.95, 0.01

# 选最优阈值的指标（用于“全局最优”）：'mcc' 或 'f1'
SELECT_METRIC = "mcc"


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

def metrics_at_threshold(y_true, y_prob, thr: float) -> Dict[str, float]:
    y_pred = (y_prob >= thr).astype(int)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)
    mcc       = matthews_corrcoef(y_true, y_pred)
    auprc     = average_precision_score(y_true, y_prob)
    auroc     = roc_auc_score(y_true, y_prob)
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
    ensure_dirs()

    # ---------------- 1) 5 折：为每个 w_pos 得到 OOF 分数 ----------------
    thr_grid = np.arange(THR_START, THR_END + 1e-9, THR_STEP)
    final_input_len = max_length_over_all_folds()

    weight_to_ys = {}  # w -> (y_all, prob_all)
    for w in W_POS_LIST:
        y_all, p_all = [], []
        print(f"[INFO] Building OOF with w_pos={w} ...")

        for k in range(1, 6):
            pos_tr, pos_te, neg_tr, neg_te = load_fold_data(k)
            X_tr_raw = list(pos_tr.values()) + list(neg_tr.values())
            y_tr     = np.array([1]*len(pos_tr) + [0]*len(neg_tr), dtype=int)

            X_te_raw = list(pos_te.values()) + list(neg_te.values())
            y_te     = np.array([1]*len(pos_te) + [0]*len(neg_te), dtype=int)

            # 统一长度
            X_tr = pad_or_truncate(X_tr_raw, final_input_len)
            X_te = pad_or_truncate(X_te_raw, final_input_len)

            # 训练 SVM（RBF），使用 cost-sensitive 权重
            cw = {0: 1.0, 1: float(w)}
            clf = SVC(kernel="rbf", C=SVM_C, gamma=SVM_GAMMA, probability=True, class_weight=cw)
            clf.fit(X_tr, y_tr)

            prob_te = clf.predict_proba(X_te)[:, 1]
            y_all.append(y_te)
            p_all.append(prob_te)

        y_all = np.concatenate(y_all)
        p_all = np.concatenate(p_all)
        weight_to_ys[w] = (y_all, p_all)

        pos_ratio = y_all.mean()
        print(f"  OOF size={len(y_all)}, pos_ratio={pos_ratio:.4f}")

    # ---------------- 2) 阈值 × w_pos 曲线 + CSV ----------------
    rows_f1, rows_mcc = [], []
    best_by_w_f1, best_by_w_mcc = [], []

    for w, (y_all, p_all) in weight_to_ys.items():
        best_f1, best_f1_row = -1, None
        best_mcc, best_mcc_row = -1, None
        for thr in thr_grid:
            mts = metrics_at_threshold(y_all, p_all, thr)
            mts["w_pos"] = w

            rows_f1.append({**mts})
            rows_mcc.append({**mts})

            if mts["f1"] > best_f1:
                best_f1, best_f1_row = mts["f1"], dict(mts)
            if mts["mcc"] > best_mcc:
                best_mcc, best_mcc_row = mts["mcc"], dict(mts)

        best_by_w_f1.append(best_f1_row)
        best_by_w_mcc.append(best_mcc_row)

    df_grid = pd.DataFrame(rows_f1)  # 含所有指标
    grid_csv = os.path.join(RESULT_DIR, "threshold_grid_all_metrics.csv")
    df_grid.to_csv(grid_csv, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {grid_csv}")

    df_best_f1  = pd.DataFrame(best_by_w_f1).sort_values("w_pos")
    df_best_mcc = pd.DataFrame(best_by_w_mcc).sort_values("w_pos")
    best_f1_csv  = os.path.join(RESULT_DIR, "best_by_weight_f1.csv")
    best_mcc_csv = os.path.join(RESULT_DIR, "best_by_weight_mcc.csv")
    df_best_f1.to_csv(best_f1_csv, index=False, encoding="utf-8-sig");  print(f"[SAVE] {best_f1_csv}")
    df_best_mcc.to_csv(best_mcc_csv, index=False, encoding="utf-8-sig"); print(f"[SAVE] {best_mcc_csv}")

    # ---------------- 3) 画曲线（F1 & MCC vs Threshold，多条线） ----------------
    import matplotlib.pyplot as plt
    plt.switch_backend("Agg")

    for metric_name in ["f1", "mcc"]:
        plt.figure(figsize=(9, 6))
        for w in W_POS_LIST:
            dfw = df_grid[df_grid["w_pos"] == w].sort_values("threshold")
            plt.plot(dfw["threshold"].values, dfw[metric_name].values, label=f"w_pos={w}")
        plt.xlabel("Threshold"); plt.ylabel(metric_name.upper())
        plt.title(f"{metric_name.upper()} vs Threshold (SVM RBF, C={SVM_C}, gamma={SVM_GAMMA})")
        plt.legend()
        out_png = os.path.join(RESULT_DIR, f"curves_{metric_name}.png")
        plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()
        print(f"[SAVE] {out_png}")

    # ---------------- 4) 选“全局最优（按 SELECT_METRIC）” ----------------
    df_pick = df_best_mcc if SELECT_METRIC.lower() == "mcc" else df_best_f1
    idx = df_pick[SELECT_METRIC.lower()].astype(float).idxmax()
    best_global = df_pick.loc[idx]
    best_w  = float(best_global["w_pos"])
    best_thr = float(best_global["threshold"])
    print(f"[INFO] GLOBAL best ({SELECT_METRIC}): w_pos={best_w}, thr={best_thr:.3f}, "
          f"F1={best_global['f1']:.4f}, MCC={best_global['mcc']:.4f}")

    # ---------------- 5) 全训练集重训（用最佳 w_pos），保存模型 ----------------
    # 复用 final_input_len 作为最终输入长度
    X_all, y_all = [], []
    for k in range(1, 6):
        pos_tr, _, neg_tr, _ = load_fold_data(k)
        X_all.extend(list(pos_tr.values()))
        X_all.extend(list(neg_tr.values()))
        y_all.extend([1]*len(pos_tr) + [0]*len(neg_tr))

    X_all = pad_or_truncate(X_all, final_input_len)
    y_all = np.array(y_all, dtype=int)

    cw = {0: 1.0, 1: float(best_w)}
    clf = SVC(kernel="rbf", C=SVM_C, gamma=SVM_GAMMA, probability=True, class_weight=cw)
    clf.fit(X_all, y_all)

    # 保存（与原路径保持一致，兼容你原 external 流程）
    import joblib
    joblib.dump({
        "model": clf,
        "input_len": final_input_len,
        "best_w_pos": best_w,
        "best_threshold": best_thr
    }, MODEL_PATH)
    print(f"[SAVE] Final model (with meta) -> {MODEL_PATH}")

    # 同步保存选择摘要
    summary = {
        "select_metric": SELECT_METRIC,
        "svm_C": SVM_C,
        "svm_gamma": SVM_GAMMA,
        "best_w_pos": best_w,
        "best_threshold": best_thr,
        "input_len": final_input_len
    }
    with open(os.path.join(RESULT_DIR, "choice_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ---------------- 6) External Test（四个 pkl，合并） ----------------
    def load_vec_list(p):
        obj = load_pickle(p)
        # 兼容 list 或 dict
        if isinstance(obj, dict):
            return list(obj.values())
        return obj

    X_pos = load_vec_list(EXTERNAL["eff_pos"]) + load_vec_list(EXTERNAL["fun_pos"])
    X_neg = load_vec_list(EXTERNAL["eff_neg"]) + load_vec_list(EXTERNAL["fun_neg"])
    print(f"[INFO] External pos={len(X_pos)}, neg={len(X_neg)}, pos_ratio={len(X_pos)/(len(X_pos)+len(X_neg)):.4f}")

    X_te_raw = X_pos + X_neg
    y_te     = np.array([1]*len(X_pos) + [0]*len(X_neg), dtype=int)
    X_te     = pad_or_truncate(X_te_raw, final_input_len)

    # 用概率 + 两个阈值（最佳阈值 & 0.5）
    proba = clf.predict_proba(X_te)[:, 1]

    def eval_with_thr(thr):
        mts = metrics_at_threshold(y_te, proba, thr)
        cm = confusion_matrix(y_te, (proba >= thr).astype(int)).tolist()
        mts["confusion_matrix"] = {"tn_fp_fn_tp_order": ["tn","fp","fn","tp"], "matrix": cm}
        return mts

    mts_best = eval_with_thr(best_thr)
    mts_050  = eval_with_thr(0.5)

    # 保存
    pd.DataFrame({
        "y_true": y_te,
        "y_score": proba,
        "y_pred_thr_best": (proba >= best_thr).astype(int),
        "y_pred_thr_0p5": (proba >= 0.5).astype(int)
    }).to_csv(os.path.join(RESULT_DIR, "external_scores_preds.csv"), index=False, encoding="utf-8-sig")

    with open(os.path.join(RESULT_DIR, "external_summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "best_threshold": best_thr,
            "metrics_at_best": mts_best,
            "metrics_at_0p5": mts_050
        }, f, ensure_ascii=False, indent=2)

    print("\n[EXTERNAL] Metrics @ best threshold")
    for k in ["precision","recall","specificity","accuracy","f1","mcc","auprc","auroc"]:
        print(f"  {k:>11}: {mts_best[k]:.4f}")
    print("[EXTERNAL] Metrics @ 0.5 threshold")
    for k in ["precision","recall","specificity","accuracy","f1","mcc","auprc","auroc"]:
        print(f"  {k:>11}: {mts_050[k]:.4f}")

    print("\n[DONE] 阈值调优（含 w_pos 曲线）+ 全训练集重训 + External Test 全流程完成。")


if __name__ == "__main__":
    main()
