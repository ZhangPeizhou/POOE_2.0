# -*- coding: utf-8 -*-
import os, pickle, warnings, numpy as np, pandas as pd
from joblib import dump
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score, matthews_corrcoef
)
warnings.filterwarnings("ignore")

import xgboost as xgb

# ==================== 基本配置 ====================
BASE_DIR   = "/content/POOE_2.0/ESM2_All_Data_5-Folds"
RESULT_DIR = os.path.join(BASE_DIR, "results_tune_xgb")
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

def gpu_available_for_xgb():
    # 粗略判断：若报错会在 fit 时自动回退
    try:
        _ = xgb.core._has_cuda_context()
        return True
    except Exception:
        return False

def make_xgb_classifier(scale_pos_weight=1.0, use_gpu=True):
    tree_method = "gpu_hist" if use_gpu else "hist"
    predictor = "gpu_predictor" if use_gpu else "auto"
    return xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",          # 以 PR-AUC 为主
        tree_method=tree_method,
        predictor=predictor,
        random_state=RANDOM_STATE,
        n_estimators=600,             # 无早停时的默认上限
        verbosity=0,
        scale_pos_weight=scale_pos_weight,
        # 其余参数由搜索空间覆盖
    )

# ==================== 超参搜索空间（随机搜索） ====================
def build_param_dist(pos_weight_candidates):
    return {
        "learning_rate": [0.02, 0.05, 0.1],
        "max_depth": [3, 5, 7, 9],
        "min_child_weight": [1, 2, 5, 10],
        "gamma": [0.0, 0.1, 0.3],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_alpha": [0.0, 1e-2, 1e-1, 1.0],
        "reg_lambda": [0.5, 1.0, 2.0, 5.0],
        "scale_pos_weight": pos_weight_candidates,  # 针对不平衡
        "n_estimators": [400, 600, 800, 1000],      # 在内层CV里先不做早停
    }

# ==================== 主流程 ====================
def main():
    use_gpu = gpu_available_for_xgb()

    # ---------- 外层5折：每折做“随机搜索”，在该折测试集上评估 ----------
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

        # 针对该折训练集的不平衡，给出候选 scale_pos_weight
        pos_cnt = (ytr == 1).sum(); neg_cnt = (ytr == 0).sum()
        ratio = (neg_cnt / max(1, pos_cnt))
        spw_cands = [1.0, 0.5*ratio, ratio, 1.5*ratio]

        xgb_base = make_xgb_classifier(use_gpu=use_gpu)
        search = RandomizedSearchCV(
            estimator=xgb_base,
            param_distributions=build_param_dist(spw_cands),
            n_iter=32,
            scoring="average_precision",
            cv=CV_INNER,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=0,
            refit=True
        )
        search.fit(Xtr, ytr)

        yprob = search.predict_proba(Xte)[:, 1]
        scores = multi_scores(yte, yprob, thr=0.5)

        row = {"Fold": k, **scores, "inner_best_cv_avg_precision": float(search.best_score_)}
        for p, v in search.best_params_.items():
            row[f"param_{p}"] = v
        per_fold_rows.append(row)

        # 保存每折预测
        pd.DataFrame({"Protein_ID": ids_test, "Label": yte, "Pred_Prob": yprob}).to_csv(
            os.path.join(RESULT_DIR, f"xgb_fold{k}_pred.csv"), index=False
        )

    df = pd.DataFrame(per_fold_rows)
    df.to_csv(os.path.join(RESULT_DIR, "xgb_outerfold_scores.csv"), index=False)
    df[["AUPRC","AUROC","Precision","Recall","F1","MCC","Accuracy","Specificity"]].agg(["mean","std"]).to_csv(
        os.path.join(RESULT_DIR, "xgb_outerfold_summary.csv")
    )

    # ---------- 全量训练：再次随机搜索 → 用早停训练最终模型 ----------
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

    pos_cnt = (y_all == 1).sum(); neg_cnt = (y_all == 0).sum()
    ratio = (neg_cnt / max(1, pos_cnt))
    spw_cands = [1.0, 0.5*ratio, ratio, 1.5*ratio]

    xgb_base = make_xgb_classifier(use_gpu=use_gpu)
    global_search = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=build_param_dist(spw_cands),
        n_iter=48,
        scoring="average_precision",
        cv=CV_INNER,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
        refit=True
    )
    global_search.fit(X_all, y_all)
    best_params = global_search.best_params_

    # 用早停训练最终模型（从全量训练中留出 10% 做 valid）
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_all, y_all, test_size=0.1, random_state=RANDOM_STATE, stratify=y_all
    )
    final_model = make_xgb_classifier(
        use_gpu=use_gpu,
        scale_pos_weight=best_params.get("scale_pos_weight", 1.0)
    )
    # 将其它最优超参写入
    for k, v in best_params.items():
        if k != "scale_pos_weight":
            setattr(final_model, k, v)

    final_model.set_params(n_estimators=max(800, best_params.get("n_estimators", 800)))  # 给早停充足上限
    final_model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
        early_stopping_rounds=100
    )

    # 保存最终模型
    model_path = os.path.join(MODEL_DIR, "xgb_final_tuned.joblib")
    dump(final_model, model_path)

    # 记录参数与摘要
    with open(os.path.join(RESULT_DIR, "xgb_selected_params.txt"), "w") as f:
        f.write(f"USE_GPU={use_gpu}\n")
        f.write("[Global best params (Randomized CV)]\n")
        f.write(str(best_params) + "\n")
        f.write(f"Global best CV (avg_precision): {float(global_search.best_score_):.6f}\n")
        f.write(f"Final feature length used: {final_len}\n")
        f.write(f"Best n_estimators used (early-stopped): {final_model.best_iteration_ if hasattr(final_model,'best_iteration_') else 'N/A'}\n")

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
        np.save(os.path.join(RESULT_DIR, "xgb_external_unbalanced_ytrue.npy"), y_ext)
        np.save(os.path.join(RESULT_DIR, "xgb_external_unbalanced_yprob.npy"), yprob_ext)
        sc = multi_scores(y_ext, yprob_ext)
        pd.DataFrame([sc]).to_csv(os.path.join(RESULT_DIR, "xgb_external_unbalanced_scores.csv"), index=False)

    # 1:3 固定外测
    objs2 = [safe_load(BAL_POS), safe_load(BAL_NEG)]
    if all(o is not None for o in objs2):
        Xp = flatten_list_or_dict(objs2[0]); Xn = flatten_list_or_dict(objs2[1])
        X_bal = pad_or_truncate(Xp + Xn, final_len)
        y_bal = np.array([1]*len(Xp) + [0]*len(Xn), dtype=int)
        yprob_bal = final_model.predict_proba(X_bal)[:, 1]
        np.save(os.path.join(RESULT_DIR, "xgb_external_1to3_ytrue.npy"), y_bal)
        np.save(os.path.join(RESULT_DIR, "xgb_external_1to3_yprob.npy"), yprob_bal)
        sc2 = multi_scores(y_bal, yprob_bal)
        pd.DataFrame([sc2]).to_csv(os.path.join(RESULT_DIR, "xgb_external_1to3_scores.csv"), index=False)

if __name__ == "__main__":
    main()
