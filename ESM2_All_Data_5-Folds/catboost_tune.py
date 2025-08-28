# -*- coding: utf-8 -*-
"""
CatBoost 调参（CPU）+ 最终训练（GPU）稳妥版
- 外层5折：CPU RandomizedSearchCV（AUPRC评分），可选SVD降维（仅搜索/或全流程）
- 全量：抽样50%再搜索（CPU），随后GPU+早停训练最终模型
- 严格限显存：gpu_ram_part=0.18，depth<=6，iterations<=1000
- 保存：每折预测/外测预测与指标/最终模型(+可选SVD)
"""

import os, pickle, warnings, numpy as np, pandas as pd, gc
from joblib import dump
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score, matthews_corrcoef
)
from sklearn.decomposition import TruncatedSVD

warnings.filterwarnings("ignore")

from catboost import CatBoostClassifier, Pool

# ==================== 配置 ====================
BASE_DIR   = "/content/POOE_2.0/ESM2_All_Data_5-Folds"
RESULT_DIR = os.path.join(BASE_DIR, "results_tune_catboost")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)

RANDOM_STATE = 42
CV_INNER = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

# --- 调参/训练策略 ---
SEARCH_ON_CPU = True          # 调参阶段用CPU（稳），最终模型用GPU
SUB_SAMPLE_FOR_SEARCH = 0.5   # 全量阶段：抽样比例做搜索（降低内存/显存）
TARGET_SVD_DIM = 0          # =0 关闭SVD；>0 则启用降维到该维（建议512/1024）。默认：512

# --- 外部测试路径（存在才评估） ---
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
    X = np.zeros((len(features_list), target_len), dtype=np.float32)
    for i, feat in enumerate(features_list):
        a = np.asarray(feat).reshape(-1).astype(np.float32)
        L = len(a)
        if L >= target_len: X[i] = a[:target_len]
        else: X[i, :L] = a
    return X

def multi_scores(y_true, y_prob, thr=0.5):
    y_pred = (y_prob >= thr).astype(int)
    TN, FP, FN, TP = confusion_matrix(y_true, y_pred).ravel()
    spec = TN/(TN+FP) if (TN+FP)>0 else 0.0
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

def maybe_svd_fit_transform(X_train, X_valid=None, target_dim=0):
    """target_dim>0 才做SVD；返回 (X_train_reduced, X_valid_reduced, svd or None)"""
    if target_dim and X_train.shape[1] > target_dim:
        svd = TruncatedSVD(n_components=target_dim, random_state=RANDOM_STATE)
        Xtr = svd.fit_transform(X_train)
        Xva = svd.transform(X_valid) if X_valid is not None else None
        return Xtr, Xva, svd
    return X_train, X_valid, None

# ==================== CatBoost 工厂（CPU / GPU） ====================
def make_cat_cpu(scale_pos_weight=1.0):
    return CatBoostClassifier(
        task_type="CPU", train_dir=None,
        loss_function="Logloss", eval_metric="AUC",
        random_seed=RANDOM_STATE,
        iterations=800, depth=6, learning_rate=0.05,
        bootstrap_type="Bernoulli", subsample=0.8, rsm=0.8,
        l2_leaf_reg=3.0, border_count=128,
        thread_count=-1, verbose=False
    )

def make_cat_gpu(scale_pos_weight=1.0):
    return CatBoostClassifier(
        task_type="GPU", devices="0", train_dir=None,
        loss_function="Logloss", eval_metric="AUC",
        random_seed=RANDOM_STATE,
        iterations=800, depth=6, learning_rate=0.05,
        bootstrap_type="Bernoulli", subsample=0.8, rsm=0.8,
        l2_leaf_reg=3.0, border_count=128,
        gpu_ram_part=0.18,  # 更严格的显存限制
        verbose=False
    )

# ==================== 主流程 ====================
def main():
    print("=== CatBoost 稳妥版：CPU 调参 + GPU 终训（含可选SVD） ===")
    print(f"配置：SEARCH_ON_CPU={SEARCH_ON_CPU}, SUB_SAMPLE_FOR_SEARCH={SUB_SAMPLE_FOR_SEARCH}, TARGET_SVD_DIM={TARGET_SVD_DIM}")

    # ---------- 外层 5 折（CPU 搜索） ----------
    per_fold = []
    for k in range(1, 6):
        print(f"\n>>> 开始第{k}折")
        pos_tr, pos_te, neg_tr, neg_te = load_fold_data(k)
        Xtr_raw = list(pos_tr.values()) + list(neg_tr.values())
        ytr = np.array([1]*len(pos_tr) + [0]*len(neg_tr), dtype=int)
        Xte_raw = list(pos_te.values()) + list(neg_te.values())
        yte = np.array([1]*len(pos_te) + [0]*len(neg_te), dtype=int)

        max_len = max(
            max(len(np.asarray(x).reshape(-1)) for x in Xtr_raw),
            max(len(np.asarray(x).reshape(-1)) for x in Xte_raw)
        )
        Xtr_full = pad_or_truncate(Xtr_raw, max_len)
        Xte_full = pad_or_truncate(Xte_raw, max_len)

        # 只在搜索阶段可选SVD（加速 & 降内存）
        Xtr, Xte, svd_cv = maybe_svd_fit_transform(Xtr_full, Xte_full, target_dim=TARGET_SVD_DIM)

        ids_test = list(pos_te.keys()) + list(neg_te.keys())
        pos_cnt = (ytr==1).sum(); neg_cnt = (ytr==0).sum()
        ratio = (neg_cnt / max(1, pos_cnt))
        spw_cands = [1.0, 0.7*ratio, ratio, 1.3*ratio]

        param_dist = {
            "learning_rate": [0.03, 0.05, 0.08],
            "depth": [4, 6],
            "l2_leaf_reg": [3.0, 6.0, 10.0],
            "subsample": [0.8, 1.0],
            "rsm": [0.6, 0.8],
            "scale_pos_weight": spw_cands,
            "iterations": [600, 800]
        }

        model_for_search = make_cat_cpu() if SEARCH_ON_CPU else make_cat_gpu()
        search = RandomizedSearchCV(
            estimator=model_for_search,
            param_distributions=param_dist,
            n_iter=16,
            scoring="average_precision",
            cv=CV_INNER,
            random_state=RANDOM_STATE,
            n_jobs=1,                   # 串行，稳
            pre_dispatch="1*n_jobs",
            refit=True
        )
        search.fit(Xtr, ytr)
        print(f"Fold{k} 内层最佳 AUPRC={search.best_score_:.4f}  参数={search.best_params_}")

        # 用最佳模型在（同一表示空间）上预测外层测试
        best_est = search.best_estimator_
        yprob = best_est.predict_proba(Xte)[:,1]
        sc = multi_scores(yte, yprob)
        print(f"Fold{k} 外层测试 AUPRC={sc['AUPRC']:.4f} AUROC={sc['AUROC']:.4f}")

        per_fold.append({"Fold":k, **sc, "inner_best_cv_avg_precision": float(search.best_score_)})

        pd.DataFrame({"Protein_ID": ids_test, "Label": yte, "Pred_Prob": yprob}).to_csv(
            os.path.join(RESULT_DIR, f"cat_fold{k}_pred.csv"), index=False
        )

        # 释放
        del search, best_est, Xtr, Xte, Xtr_full, Xte_full; gc.collect()

    df = pd.DataFrame(per_fold)
    df.to_csv(os.path.join(RESULT_DIR, "cat_outerfold_scores.csv"), index=False)
    print("\n=== 外层 5 折完成 ===")
    print(df[["Fold","AUPRC","AUROC","Precision","Recall","F1"]])

    # ---------- 全量：抽样搜索（CPU） → 最终训练（GPU + 早停） ----------
    print("\n>>> 开始全量阶段：抽样搜索（CPU）+ GPU 终训（早停）")
    final_len = 0; all_pos, all_neg = [], []
    for k in range(1, 6):
        pos_tr, _, neg_tr, _ = load_fold_data(k)
        all_pos.extend(list(pos_tr.values())); all_neg.extend(list(neg_tr.values()))
        lens = [len(np.asarray(v).reshape(-1)) for v in list(pos_tr.values()) + list(neg_tr.values())]
        final_len = max(final_len, max(lens))
    X_all_full = pad_or_truncate(all_pos + all_neg, final_len)
    y_all = np.array([1]*len(all_pos) + [0]*len(all_neg), dtype=int)

    # —— 抽样（分层）用于搜索 —— #
    rng = np.random.RandomState(RANDOM_STATE)
    idx_pos = np.where(y_all==1)[0]; idx_neg = np.where(y_all==0)[0]
    sp = rng.choice(idx_pos, max(1, int(len(idx_pos)*SUB_SAMPLE_FOR_SEARCH)), replace=False)
    sn = rng.choice(idx_neg, max(1, int(len(idx_neg)*SUB_SAMPLE_FOR_SEARCH)), replace=False)
    sub_idx = np.concatenate([sp, sn])
    X_sub_full, y_sub = X_all_full[sub_idx], y_all[sub_idx]

    # 搜索阶段也可选SVD
    X_sub, _, svd_global = maybe_svd_fit_transform(X_sub_full, None, target_dim=TARGET_SVD_DIM)

    pos_cnt = (y_sub==1).sum(); neg_cnt = (y_sub==0).sum()
    ratio = (neg_cnt / max(1, pos_cnt))
    spw_cands = [1.0, 0.7*ratio, ratio, 1.3*ratio]
    param_dist2 = {
        "learning_rate": [0.03, 0.05, 0.08],
        "depth": [4, 6],
        "l2_leaf_reg": [3.0, 6.0, 10.0],
        "subsample": [0.8, 1.0],
        "rsm": [0.6, 0.8],
        "scale_pos_weight": spw_cands,
        "iterations": [600, 800]
    }

    search2 = RandomizedSearchCV(
        estimator=make_cat_cpu() if SEARCH_ON_CPU else make_cat_gpu(),
        param_distributions=param_dist2,
        n_iter=24,
        scoring="average_precision",
        cv=CV_INNER,
        random_state=RANDOM_STATE,
        n_jobs=1,
        pre_dispatch="1*n_jobs",
        refit=True
    )
    search2.fit(X_sub, y_sub)
    best_params = search2.best_params_
    print(f"全量(抽样)随机搜索最佳 AUPRC={search2.best_score_:.4f}  参数={best_params}")

    # —— 最终训练：GPU + 早停（可选：是否也对全量做同样的SVD） —— #
    APPLY_SVD_TO_FINAL = TARGET_SVD_DIM > 0  # 为保证一致性，若搜索用了SVD，这里也应用同一维度
    if APPLY_SVD_TO_FINAL:
        X_all, _, svd_final = maybe_svd_fit_transform(X_all_full, None, target_dim=TARGET_SVD_DIM)
        dump(svd_final, os.path.join(MODEL_DIR, "catboost_svd.joblib"))
        svd_used = svd_final
    else:
        X_all = X_all_full
        svd_used = None

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_all, y_all, test_size=0.1, random_state=RANDOM_STATE, stratify=y_all
    )

    final_model = make_cat_gpu(scale_pos_weight=best_params.get("scale_pos_weight", 1.0))
    for k,v in best_params.items():
        final_model.set_params(**{k: v})
    if final_model.get_param("iterations") < 1000:
        final_model.set_params(iterations=1000)  # 给早停空间

    final_model.fit(
        Pool(X_tr, y_tr),
        eval_set=Pool(X_val, y_val),
        use_best_model=True,
        verbose=False
    )
    print("最终模型已训练（GPU），best_iterations 可能小于上限（早停）")

    model_path = os.path.join(MODEL_DIR, "catboost_final_tuned.joblib")
    dump(final_model, model_path)
    with open(os.path.join(RESULT_DIR, "cat_final_params.txt"), "w") as f:
        f.write(f"[Best params from search]\n{best_params}\n")
        f.write(f"SEARCH_ON_CPU={SEARCH_ON_CPU}, SUB_SAMPLE_FOR_SEARCH={SUB_SAMPLE_FOR_SEARCH}, TARGET_SVD_DIM={TARGET_SVD_DIM}\n")
    print(f"最终模型已保存：{model_path}")
    if svd_used is not None:
        print("SVD 也已保存：models/catboost_svd.joblib")

    # ---------- 外部测试（未配平/1:3），保存 y_true/y_prob ----------
    def safe_load(path):
        try:
            with open(path, "rb") as f: return pickle.load(f)
        except Exception: return None

    def eval_external(tag, pos_paths, neg_paths):
        pos_objs, neg_objs = [], []
        for p in pos_paths:
            o = safe_load(p); 
            if o is not None: pos_objs.extend(flatten_list_or_dict(o))
        for p in neg_paths:
            o = safe_load(p);
            if o is not None: neg_objs.extend(flatten_list_or_dict(o))
        if not pos_objs or not neg_objs: 
            return

        X_ext_full = pad_or_truncate(pos_objs + neg_objs, final_len)
        y_ext = np.array([1]*len(pos_objs) + [0]*len(neg_objs), dtype=int)

        # 若最终模型用了 SVD，则对外测也做同样变换
        if svd_used is not None:
            X_ext = svd_used.transform(X_ext_full)
        else:
            X_ext = X_ext_full

        yprob = final_model.predict_proba(X_ext)[:,1]
        sc = multi_scores(y_ext, yprob)
        print(f"\n外部测试 {tag}: Precision={sc['Precision']:.3f} Recall={sc['Recall']:.3f} AUPRC={sc['AUPRC']:.3f}")

        np.save(os.path.join(RESULT_DIR, f"cat_external_{tag}_ytrue.npy"), y_ext)
        np.save(os.path.join(RESULT_DIR, f"cat_external_{tag}_yprob.npy"), yprob)
        pd.DataFrame([sc]).to_csv(os.path.join(RESULT_DIR, f"cat_external_{tag}_scores.csv"), index=False)

    eval_external("unbalanced", [EFF_POS, FUN_POS], [EFF_NEG, FUN_NEG])
    eval_external("1to3", [BAL_POS], [BAL_NEG])

if __name__ == "__main__":
    main()
