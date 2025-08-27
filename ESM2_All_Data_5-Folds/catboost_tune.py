# -*- coding: utf-8 -*-
import os, pickle, warnings, numpy as np, pandas as pd
from joblib import dump
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score, matthews_corrcoef
)
warnings.filterwarnings("ignore")

from catboost import CatBoostClassifier, Pool

# ==================== 基本配置 ====================
BASE_DIR   = "/content/POOE_2.0/ESM2_All_Data_5-Folds"
RESULT_DIR = os.path.join(BASE_DIR, "results_tune_catboost")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)

RANDOM_STATE = 42
CV_INNER = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

# 外部测试路径（存在才会评估）
EFF_POS = "/content/POOE_2.0/EffectorP-3.0-Data/TestData_Embedding_ESM2/positivedata_external_test.pkl"
EFF_NEG = "/content/POOE_2.0/EffectorP-3.0-Data/TestData_Embedding_ESM2/negativedata_external_test.pkl"
FUN_POS = "/content/POOE_2.0/Fungtion-Data/Fungtion_Independent_Embedding_ESM2/positivedata_fungtion.pkl"
FUN_NEG = "/content/POOE_2.0/Fungtion-Data/Fungtion_Independent_Embedding_ESM2/negativedata_fungtion.pkl"
BAL_POS = "/content/POOE_2.0/External_Test_1to3_Balanced/positivedata_test_balanced.pkl"
BAL_NEG = "/content/POOE_2.0/External_Test_1to3_Balanced/negativedata_test_balanced.pkl"

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

def make_cat_model(scale_pos_weight=1.0):
    # GPU 固定启用；AUPRC 在外侧用 sklearn 计算
    return CatBoostClassifier(
        task_type="GPU", devices="0",
        loss_function="Logloss", eval_metric="AUC",
        random_seed=RANDOM_STATE,
        iterations=1000,              # CV 阶段不早停
        learning_rate=0.05, depth=8,
        l2_leaf_reg=3.0, subsample=0.8, rsm=0.8,
        scale_pos_weight=scale_pos_weight,
        verbose=False
    )

# ==================== 主流程 ====================
def main():
    print("=== CatBoost (GPU) 调参开始 ===")

    # ---------- 外层5折 ----------
    per_fold_rows = []
    for k in range(1, 6):
        print(f"\n>>> 开始第{k}折")
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

        # 不平衡：给 scale_pos_weight 候选
        pos_cnt = (ytr == 1).sum(); neg_cnt = (ytr == 0).sum()
        ratio = (neg_cnt / max(1, pos_cnt))
        spw_cands = [1.0, 0.5*ratio, ratio, 1.5*ratio]

        # 随机搜索空间
        param_dist = {
            "learning_rate": [0.02, 0.05, 0.1],
            "depth": [6, 8, 10],
            "l2_leaf_reg": [1.0, 3.0, 5.0, 10.0],
            "subsample": [0.7, 0.8, 1.0],
            "rsm": [0.6, 0.8, 1.0],
            "scale_pos_weight": spw_cands,
            "iterations": [800, 1000, 1400]
        }

        search = RandomizedSearchCV(
            estimator=make_cat_model(),
            param_distributions=param_dist,
            n_iter=20,
            scoring="average_precision",
            cv=CV_INNER,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            refit=True
        )
        search.fit(Xtr, ytr)
        print(f"Fold{k} 内层最佳 AUPRC={search.best_score_:.4f}, 参数={search.best_params_}")

        best_est = search.best_estimator_
        yprob = best_est.predict_proba(Xte)[:, 1]
        scores = multi_scores(yte, yprob)
        print(f"Fold{k} 外层测试集 AUPRC={scores['AUPRC']:.4f}, AUROC={scores['AUROC']:.4f}")

        row = {"Fold": k, **scores, "inner_best_cv_avg_precision": float(search.best_score_)}
        per_fold_rows.append(row)

        pd.DataFrame({"Protein_ID": ids_test, "Label": yte, "Pred_Prob": yprob}).to_csv(
            os.path.join(RESULT_DIR, f"cat_fold{k}_pred.csv"), index=False
        )

    df = pd.DataFrame(per_fold_rows)
    df.to_csv(os.path.join(RESULT_DIR, "cat_outerfold_scores.csv"), index=False)
    print("\n=== 外层5折完成 ===")
    print(df[["Fold","AUPRC","AUROC","Precision","Recall","F1"]])

    # ---------- 全量训练：再次随机搜索 → Early Stopping ----------
    print("\n>>> 开始全量训练 (随机搜索+早停)")
    final_len = 0
    all_pos, all_neg = [], []
    for k in range(1, 6):
        pos_tr, _, neg_tr, _ = load_fold_data(k)
        all_pos.extend(list(pos_tr.values())); all_neg.extend(list(neg_tr.values()))
        lens = [len(np.asarray(v).flatten()) for v in list(pos_tr.values()) + list(neg_tr.values())]
        final_len = max(final_len, max(lens))
    X_all = pad_or_truncate(all_pos + all_neg, final_len)
    y_all = np.array([1]*len(all_pos) + [0]*len(all_neg), dtype=int)

    pos_cnt = (y_all == 1).sum(); neg_cnt = (y_all == 0).sum()
    ratio = (neg_cnt / max(1, pos_cnt))
    spw_cands = [1.0, 0.5*ratio, ratio, 1.5*ratio]

    param_dist["scale_pos_weight"] = spw_cands
    global_search = RandomizedSearchCV(
        estimator=make_cat_model(),
        param_distributions=param_dist,
        n_iter=30,
        scoring="average_precision",
        cv=CV_INNER,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True
    )
    global_search.fit(X_all, y_all)
    best_params = global_search.best_params_
    print(f"全量随机搜索最佳 AUPRC={global_search.best_score_:.4f}, 参数={best_params}")

    # 早停（从全量里划 10% 做验证）
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_all, y_all, test_size=0.1, random_state=RANDOM_STATE, stratify=y_all
    )
    final_model = make_cat_model(scale_pos_weight=best_params.get("scale_pos_weight", 1.0))
    for k,v in best_params.items():
        final_model.set_params(**{k: v})
    # 给早停留空间
    if final_model.get_param("iterations") < 1400:
        final_model.set_params(iterations=1400)

    final_model.fit(
        Pool(X_tr, y_tr),
        eval_set=Pool(X_val, y_val),
        use_best_model=True,
        verbose=False
    )
    print("最终模型 task_type:", final_model.get_param("task_type"))

    model_path = os.path.join(MODEL_DIR, "catboost_final_tuned.joblib")
    dump(final_model, model_path)
    print(f"最终模型已保存: {model_path}")

    # ---------- 外部测试（未配平 & 1:3），保存 y_true/y_prob ----------
    def safe_load(path):
        try:
            with open(path, "rb") as f: return pickle.load(f)
        except Exception:
            return None

    for tag, pos_paths, neg_paths in [
        ("unbalanced", [EFF_POS, FUN_POS], [EFF_NEG, FUN_NEG]),
        ("1to3", [BAL_POS], [BAL_NEG]),
    ]:
        pos_objs, neg_objs = [], []
        for p in pos_paths:
            o = safe_load(p)
            if o is not None: pos_objs.extend(flatten_list_or_dict(o))
        for p in neg_paths:
            o = safe_load(p)
            if o is not None: neg_objs.extend(flatten_list_or_dict(o))
        if not pos_objs or not neg_objs:
            continue
        X_ext = pad_or_truncate(pos_objs + neg_objs, final_len)
        y_ext = np.array([1]*len(pos_objs) + [0]*len(neg_objs), dtype=int)

        yprob_ext = final_model.predict_proba(X_ext)[:,1]
        sc = multi_scores(y_ext, yprob_ext)
        print(f"\n外部测试 {tag}: Precision={sc['Precision']:.3f}, Recall={sc['Recall']:.3f}, AUPRC={sc['AUPRC']:.3f}")

        np.save(os.path.join(RESULT_DIR, f"cat_external_{tag}_ytrue.npy"), y_ext)
        np.save(os.path.join(RESULT_DIR, f"cat_external_{tag}_yprob.npy"), yprob_ext)
        pd.DataFrame([sc]).to_csv(os.path.join(RESULT_DIR, f"cat_external_{tag}_scores.csv"), index=False)

if __name__ == "__main__":
    main()
