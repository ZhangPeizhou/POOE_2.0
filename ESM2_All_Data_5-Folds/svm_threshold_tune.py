#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from sklearn.svm import SVC
from sklearn.metrics import (
    precision_score, recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, roc_auc_score, average_precision_score
)
import matplotlib.pyplot as plt
plt.switch_backend("Agg")

# ======================== 顶部常量（按需改；不改也能跑） ========================
# 和你现有工程一致的目录结构（fold{n}_pkl/positivedata_k{n}.pkl 等）
BASE_DIR    = "/content/POOE_2.0/ESM2_All_Data_5-Folds"
RESULT_DIR  = os.path.join(BASE_DIR, "results_threshold_only")
MODEL_DIR   = os.path.join(BASE_DIR, "models")

# 原始模型路径（保持不动，继续保存“纯 SVC”以保持兼容）
MODEL_RAW_PATH = os.path.join(MODEL_DIR, "svm_final.joblib")
# 新的“带阈值”的模型包（不会覆盖上面那个）
MODEL_WITH_THR = os.path.join(MODEL_DIR, "svm_final_with_threshold.joblib")
BEST_THR_JSON  = os.path.join(RESULT_DIR, "best_threshold.json")

# 外测四个 pkl（如不存在会自动跳过外测评估）
EXTERNAL = {
    "eff_pos": "/content/POOE_2.0/EffectorP-3.0-Data/TestData_Embedding_ESM2/positivedata_external_test.pkl",
    "eff_neg": "/content/POOE_2.0/EffectorP-3.0-Data/TestData_Embedding_ESM2/negativedata_external_test.pkl",
    "fun_pos": "/content/POOE_2.0/Fungtion-Data/Fungtion_Independent_Embedding_ESM2/positivedata_fungtion.pkl",
    "fun_neg": "/content/POOE_2.0/Fungtion-Data/Fungtion_Independent_Embedding_ESM2/negativedata_fungtion.pkl",
}

# 固定为“原来的普通 SVM 参数”
SVM_C = 10.0
SVM_GAMMA = 0.25

# 阈值扫描设置（在 OOF 分数的分位区间上取点，更稳；不需要 0~1）
PCT_LOW, PCT_HIGH = 1.0, 99.0
N_THR = 200

# 选“最佳阈值”的指标：'mcc' 或 'f1'
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

def metrics_from_scores(y_true, scores, thr: float) -> Dict[str, float]:
    # 注意：这里用 decision_function 分数，默认阈值是 0.0；我们扫描 thr。
    y_pred = (scores >= thr).astype(int)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)
    mcc       = matthews_corrcoef(y_true, y_pred)
    # 下列两个与阈值无关，但一起输出便于参考
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
    ensure_dirs()

    # 1) 构造 OOF 分数（5 折，每折训练→在该折测试集上取 decision_function）
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

        clf = SVC(kernel="rbf", C=SVM_C, gamma=SVM_GAMMA, probability=False)
        clf.fit(X_tr, y_tr)

        # 用 decision_function（不启用概率）
        s_te = clf.decision_function(X_te)

        y_all.append(y_te)
        s_all.append(s_te)

    y_all = np.concatenate(y_all)
    s_all = np.concatenate(s_all)
    print(f"[INFO] OOF size={len(y_all)}, pos_ratio={y_all.mean():.4f}")

    # 2) 在 OOF 分数的分位区间内扫描阈值
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
    grid_csv = os.path.join(RESULT_DIR, "threshold_grid.csv")
    df_grid.to_csv(grid_csv, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {grid_csv}")

    # 3) 画曲线（F1 与 MCC 各一张）
    for metric_name in ["f1", "mcc"]:
        plt.figure(figsize=(9, 6))
        plt.plot(df_grid["threshold"].values, df_grid[metric_name].values)
        plt.xlabel("Threshold (decision_function)")
        plt.ylabel(metric_name.upper())
        plt.title(f"{metric_name.upper()} vs Threshold (SVM RBF, C={SVM_C}, gamma={SVM_GAMMA})")
        out_png = os.path.join(RESULT_DIR, f"curve_{metric_name}.png")
        plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()
        print(f"[SAVE] {out_png}")

    best_thr = float(best_row["threshold"])
    print(f"[INFO] Best threshold by {SELECT_METRIC}: {best_thr:.6f}")
    print("[INFO] Metrics @ best threshold:")
    print({k: round(best_row[k], 4) for k in ["precision","recall","specificity","accuracy","f1","mcc","auprc","auroc"]})

    # 4) 用所有训练数据重训“原始模型参数”的 SVM，并保存两个文件：
    #    - 原始模型（纯 SVC）：svm_final.joblib（保持兼容）
    #    - 带阈值的包：svm_final_with_threshold.joblib（含模型与 best_thr 与 input_len）
    X_all, y_all_tr = [], []
    for k in range(1, 6):
        pos_tr, _, neg_tr, _ = load_fold_data(k)
        X_all.extend(list(pos_tr.values()))
        X_all.extend(list(neg_tr.values()))
        y_all_tr.extend([1]*len(pos_tr) + [0]*len(neg_tr))
    X_all = pad_or_truncate(X_all, final_input_len)
    y_all_tr = np.array(y_all_tr, dtype=int)

    final_clf = SVC(kernel="rbf", C=SVM_C, gamma=SVM_GAMMA, probability=False)
    final_clf.fit(X_all, y_all_tr)

    import joblib
    # a) 继续保存“纯模型”，保持你原 external 流程的兼容性
    joblib.dump(final_clf, MODEL_RAW_PATH)
    print(f"[SAVE] {MODEL_RAW_PATH}")

    # b) 另存“带阈值”的模型包（推荐你以这个为准）
    joblib.dump({
        "model": final_clf,
        "threshold": best_thr,
        "score_type": "decision_function",
        "input_len": final_input_len,
        "svm_params": {"C": SVM_C, "gamma": SVM_GAMMA}
    }, MODEL_WITH_THR)
    print(f"[SAVE] {MODEL_WITH_THR}")

    with open(BEST_THR_JSON, "w", encoding="utf-8") as f:
        json.dump({"best_threshold": best_thr, "metric": SELECT_METRIC}, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] {BEST_THR_JSON}")

    # 5) External Test（如果四个 pkl 都存在就跑，否则跳过）
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
        mts_0    = eval_at_thr(s_te, 0.0)  # 默认阈值

        # 保存 external 原始分数与两种阈值的预测
        pd.DataFrame({
            "y_true": y_te,
            "score": s_te,
            "pred_thr_best": (s_te >= best_thr).astype(int),
            "pred_thr_0": (s_te >= 0.0).astype(int)
        }).to_csv(os.path.join(RESULT_DIR, "external_scores_preds.csv"), index=False, encoding="utf-8-sig")

        with open(os.path.join(RESULT_DIR, "external_summary.json"), "w", encoding="utf-8") as f:
            json.dump({
                "best_threshold": best_thr,
                "metrics_at_best": {k: float(v) for k, v in mts_best.items()},
                "metrics_at_0": {k: float(v) for k, v in mts_0.items()}
            }, f, ensure_ascii=False, indent=2)

        print("\n[EXTERNAL] Metrics @ best threshold")
        for k in ["precision","recall","specificity","accuracy","f1","mcc","auroc","auprc"]:
            print(f"  {k:>11}: {mts_best[k]:.4f}")
        print("[EXTERNAL] Metrics @ 0.0 threshold (default)")
        for k in ["precision","recall","specificity","accuracy","f1","mcc","auroc","auprc"]:
            print(f"  {k:>11}: {mts_0[k]:.4f}")

    except FileNotFoundError as e:
        print(f"[INFO] External test skipped (missing file: {e})")

    print("\n[DONE] 仅阈值调优 → 曲线 → 选最优阈值 → 保存新模型（含阈值） 已完成。")


if __name__ == "__main__":
    main()
