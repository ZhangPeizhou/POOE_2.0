# -*- coding: utf-8 -*-
import os, pickle, warnings, numpy as np, pandas as pd
from joblib import dump
from sklearn.model_selection import (
    StratifiedKFold, RandomizedSearchCV, GridSearchCV
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score, matthews_corrcoef
)
warnings.filterwarnings("ignore")

# =============== 可选启用 Halving（若 sklearn 版本支持） ===============
try:
    from sklearn.experimental import enable_halving_search_cv  # noqa: F401
    from sklearn.model_selection import HalvingRandomSearchCV, HalvingGridSearchCV
    HALVING_OK = True
except Exception:
    HALVING_OK = False

# ==================== 基本配置 ====================
BASE_DIR   = "/content/POOE_2.0/ESM2_All_Data_5-Folds"
RESULT_DIR = os.path.join(BASE_DIR, "results_tune_rf_coarsefine")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)

RANDOM_STATE = 42
CV_INNER = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

# 外部测试路径（若不存在会自动跳过）
EFF_POS = "/content/POOE_2.0/EffectorP-3.0-Data/TestData_Embedding_ESM2/positivedata_external_test.pkl"
EFF_NEG = "/content/POOE_2.0/EffectorP-3.0-Data/TestData_Embedding_ESM2/negativedata_external_test.pkl"
FUN_POS = "/content/POOE_2.0/Fungtion-Data/Fungtion_Independent_Embedding_ESM2/positivedata_fungtion.pkl"
FUN_NEG = "/content/POOE_2.0/Fungtion-Data/Fungtion_Independent_Embedding_ESM2/negativedata_fungtion.pkl"
BAL_POS = "/content/POOE_2.0/External_Test_1to3_Balanced/positivedata_test_balanced.pkl"
BAL_NEG = "/content/POOE_2.0/External_Test_1to3_Balanced/negativedata_test_balanced.pkl"

# =============== 选择 RF 实现：GPU（cuML）优先，失败回退 sklearn ===============
USING_CUML = False
try:
    from cuml.ensemble import RandomForestClassifier as cuRF
    RFClass = cuRF
    USING_CUML = True
except Exception:
    from sklearn.ensemble import RandomForestClassifier as skRF
    RFClass = skRF
    USING_CUML = False

# ==================== 工具函数 ====================
def flatten_list_or_dict(obj):
    if isinstance(obj, dict): vecs = list(obj.values())
    else: vecs = list(obj)
    out = []
    for v in vecs:
        a = np.asarray(v)
        if a.ndim > 1: a = a.flatten()
        out.append(a.astype(np.float32))
    return out

def load_fold_data(fold_num):
    p = os.path.join(BASE_DIR, f"fold{fold_num}_pkl")
    with open(os.path.join(p, f"positivedata_k{fold_num}.pkl"), "rb") as f: pos_tr = pickle.load(f)
    with open(os.path.join(p, f"positivedata_test_k{fold_num}.pkl"), "rb") as f: pos_te = pickle.load(f)
    with open(os.path.join(p, f"negativedata_k{fold_num}.pkl"), "rb") as f: neg_tr = pickle.load(f)
    with open(os.path.join(p, f"negativedata_test_k{fold_num}.pkl"), "rb") as f: neg_te = pickle.load(f)
    return pos_tr, pos_te, neg_tr, neg_te

def pad_or_truncate(features_list, target_len):
    fixed = np.zeros((len(features_list), target_len), dtype=np.float32)
    for i, feat in enumerate(features_list):
        arr = np.asarray(feat)
        if arr.ndim > 1: arr = arr.flatten()
        L = len(arr)
        if L >= target_len: fixed[i] = arr[:target_len]
        else: fixed[i, :L] = arr
    return fixed

def multi_scores(y_true, y_prob, thr=0.5):
    y_pred = (y_prob >= thr).astype(int)
    TN, FP, FN, TP = confusion_matrix(y_true, y_pred).ravel()
    spec = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    return dict(
        Precision=precision_score(y_true, y_pred, zero_division=0),
        Recall=recall_score(y_true, y_pred, zero_division=0),
        Specificity=spec,
        Accuracy=accuracy_score(y_true, y_pred),
        F1=f1_score(y_true, y_pred, zero_division=0),
        MCC=matthews_corrcoef(y_true, y_pred),
        AUROC=roc_auc_score(y_true, y_prob),
        AUPRC=average_precision_score(y_true, y_prob),
    )

# ==================== 调参器：粗搜 & 精网格 ====================
def make_base_rf():
    if USING_CUML:
        # cuML RF (GPU)
        return RFClass(
            random_state=RANDOM_STATE,
            n_estimators=200
        )
    else:
        # sklearn RF (CPU)
        return RFClass(
            random_state=RANDOM_STATE,
            n_estimators=200,
            n_jobs=-1
        )

def coarse_search():
    # 粗搜空间：覆盖广一点
    param_distributions = {
        "n_estimators": [200, 400, 600, 800, 1000],
        "max_depth": [None, 10, 20, 40],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", 1.0],
    }
    base = make_base_rf()
    if HALVING_OK:
        return HalvingRandomSearchCV(
            estimator=base,
            param_distributions=param_distributions,
            factor=3,                       # 每轮保留 1/factor
            resource="n_estimators",        # 用树数作为“资源”
            min_resources=200,
            max_resources=1000,
            scoring="average_precision",
            cv=CV_INNER,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=0,
            refit=True
        )
    else:
        return RandomizedSearchCV(
            estimator=base,
            param_distributions=param_distributions,
            n_iter=32,
            scoring="average_precision",
            cv=CV_INNER,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=0,
            refit=True
        )

def build_small_grid(around_best: dict):
    # 围绕 coarse 的 best_params 做“小网格”
    def neigh(val, candidates):
        # 如果 val 在候选里就用原集合；否则把 val 合并进去
        vals = list(candidates)
        if val not in vals:
            vals.append(val)
        # 去重后排序（混合类型时保持原顺序）
        try:
            return sorted(set(vals))
        except Exception:
            return list(dict.fromkeys(vals))

    grid = {
        "n_estimators":  neigh(around_best.get("n_estimators", 600), [400, 600, 800, 1000]),
        "max_depth":     neigh(around_best.get("max_depth", 20),     [None, 10, 20, 40]),
        "min_samples_split": neigh(around_best.get("min_samples_split", 2), [2, 5, 10]),
        "min_samples_leaf":  neigh(around_best.get("min_samples_leaf", 1),  [1, 2, 4]),
        "max_features":  neigh(around_best.get("max_features", "sqrt"), ["sqrt", "log2", 1.0]),
    }
    return grid

def fine_search(small_grid):
    base = RFClass(
        random_state=RANDOM_STATE,
        n_streams=1 if USING_CUML else None,
        n_jobs=-1 if not USING_CUML else None
    )
    if HALVING_OK:
        return HalvingGridSearchCV(
            estimator=base,
            param_grid=small_grid,
            factor=3,
            resource="n_estimators",
            scoring="average_precision",
            cv=CV_INNER,
            n_jobs=-1,
            verbose=0,
            refit=True
        )
    else:
        return GridSearchCV(
            estimator=base,
            param_grid=small_grid,
            scoring="average_precision",
            cv=CV_INNER,
            n_jobs=-1,
            verbose=0,
            refit=True
        )

# ==================== 主流程 ====================
def main():
    # ---------- 外层5折：每折做“粗搜”，在该折测试集上评估 ----------
    per_fold_rows = []
    for k in range(1, 6):
        pos_tr, pos_te, neg_tr, neg_te = load_fold_data(k)
        Xtr_raw = list(pos_tr.values()) + list(neg_tr.values())
        ytr = np.array([1]*len(pos_tr) + [0]*len(neg_tr), dtype=int)
        Xte_raw = list(pos_te.values()) + list(neg_te.values())
        yte = np.array([1]*len(pos_te) + [0]*len(neg_te), dtype=int)

        max_len = max(max(len(np.asarray(x).flatten()) for x in Xtr_raw),
                      max(len(np.asarray(x).flatten()) for x in Xte_raw))
        Xtr = pad_or_truncate(Xtr_raw, max_len)
        Xte = pad_or_truncate(Xte_raw, max_len)
        ids_test = list(pos_te.keys()) + list(neg_te.keys())

        coarse = coarse_search()
        coarse.fit(Xtr, ytr)

        yprob = coarse.predict_proba(Xte)[:, 1]
        scores = multi_scores(yte, yprob, thr=0.5)

        row = {"Fold": k, **scores, "inner_best_cv_avg_precision": float(coarse.best_score_)}
        for p, v in coarse.best_params_.items():
            row[f"param_{p}"] = v
        per_fold_rows.append(row)

        # 保存每折预测
        pd.DataFrame({"Protein_ID": ids_test, "Label": yte, "Pred_Prob": yprob}).to_csv(
            os.path.join(RESULT_DIR, f"rf_coarse_fold{k}_pred.csv"), index=False
        )

    df = pd.DataFrame(per_fold_rows)
    df.to_csv(os.path.join(RESULT_DIR, "rf_outerfold_scores_coarse.csv"), index=False)
    df[["AUPRC","AUROC","Precision","Recall","F1","MCC","Accuracy","Specificity"]].agg(["mean","std"]).to_csv(
        os.path.join(RESULT_DIR, "rf_outerfold_summary_coarse.csv")
    )

    # ---------- 全量训练：粗搜 → 精网格，得到最终模型 ----------
    # 组装全量训练集 & 特征长度
    final_len = 0
    all_pos, all_neg = [], []
    for k in range(1, 6):
        pos_tr, _, neg_tr, _ = load_fold_data(k)
        all_pos.extend(list(pos_tr.values())); all_neg.extend(list(neg_tr.values()))
        lens = [len(np.asarray(v).flatten()) for v in list(pos_tr.values()) + list(neg_tr.values())]
        final_len = max(final_len, max(lens))
    X_all = pad_or_truncate(all_pos + all_neg, final_len)
    y_all = np.array([1]*len(all_pos) + [0]*len(all_neg), dtype=int)

    # 粗搜
    coarse_all = coarse_search()
    coarse_all.fit(X_all, y_all)
    best_coarse = coarse_all.best_params_

    # 小网格
    grid = build_small_grid(best_coarse)
    fine = fine_search(grid)
    fine.fit(X_all, y_all)

    final_model = fine.best_estimator_
    model_path = os.path.join(MODEL_DIR, "rf_final_tuned.joblib")
    dump(final_model, model_path)

    # 保存参数与摘要
    with open(os.path.join(RESULT_DIR, "rf_selected_params_coarse_fine.txt"), "w") as f:
        f.write(f"USING_CUML(GPU)={USING_CUML}\n\n")
        f.write("[Coarse best params]\n")
        f.write(str(best_coarse) + "\n\n")
        f.write("[Fine best params]\n")
        f.write(str(fine.best_params_) + "\n")
        f.write(f"Fine best CV (avg_precision): {float(fine.best_score_):.6f}\n")
        f.write(f"Final feature length used: {final_len}\n")

    # ---------- 可选：外部测试（未配平 & 1:3固定），保存 y_true/y_prob ----------
    def safe_load(path):
        try:
            with open(path, "rb") as f: return pickle.load(f)
        except Exception:
            return None

    # 未配平外测
    objs = [safe_load(EFF_POS), safe_load(EFF_NEG), safe_load(FUN_POS), safe_load(FUN_NEG)]
    if all(o is not None for o in objs):
        X_pos = flatten_list_or_dict(objs[0]) + flatten_list_or_dict(objs[2])
        X_neg = flatten_list_or_dict(objs[1]) + flatten_list_or_dict(objs[3])
        X_ext = pad_or_truncate(X_pos + X_neg, final_len)
        y_ext = np.array([1]*len(X_pos) + [0]*len(X_neg), dtype=int)
        yprob_ext = final_model.predict_proba(X_ext)[:, 1]
        np.save(os.path.join(RESULT_DIR, "rf_external_unbalanced_ytrue.npy"), y_ext)
        np.save(os.path.join(RESULT_DIR, "rf_external_unbalanced_yprob.npy"), yprob_ext)
        sc = multi_scores(y_ext, yprob_ext)
        pd.DataFrame([sc]).to_csv(os.path.join(RESULT_DIR, "rf_external_unbalanced_scores.csv"), index=False)

    # 1:3 固定外测
    objs2 = [safe_load(BAL_POS), safe_load(BAL_NEG)]
    if all(o is not None for o in objs2):
        Xp = flatten_list_or_dict(objs2[0]); Xn = flatten_list_or_dict(objs2[1])
        X_bal = pad_or_truncate(Xp + Xn, final_len)
        y_bal = np.array([1]*len(Xp) + [0]*len(Xn), dtype=int)
        yprob_bal = final_model.predict_proba(X_bal)[:, 1]
        np.save(os.path.join(RESULT_DIR, "rf_external_1to3_ytrue.npy"), y_bal)
        np.save(os.path.join(RESULT_DIR, "rf_external_1to3_yprob.npy"), yprob_bal)
        sc2 = multi_scores(y_bal, yprob_bal)
        pd.DataFrame([sc2]).to_csv(os.path.join(RESULT_DIR, "rf_external_1to3_scores.csv"), index=False)

if __name__ == "__main__":
    main()
