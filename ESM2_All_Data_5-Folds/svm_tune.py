import os
import pickle
import numpy as np
import pandas as pd
from joblib import dump
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score
)
from scipy.stats import loguniform
import warnings
warnings.filterwarnings("ignore")

# ======================== 配置（与现有脚本保持一致） ========================
BASE_DIR = "/content/POOE_2.0/ESM2_All_Data_5-Folds"
RESULT_DIR = os.path.join(BASE_DIR, "results_tune")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
RANDOM_STATE = 42

# ======================== 工具函数（复用你现有思路） ========================
def multi_scores(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    TN, FP, FN, TP = confusion_matrix(y_true, y_pred).ravel()
    PPV = precision_score(y_true, y_pred, zero_division=0)
    TPR = recall_score(y_true, y_pred, zero_division=0)
    TNR = TN / (TN + FP) if (TN + FP) > 0 else 0
    Acc = accuracy_score(y_true, y_pred)
    F1 = f1_score(y_true, y_pred, zero_division=0)
    denominator = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    MCC = (TP * TN - FP * FN) / denominator if denominator != 0 else 0
    AUROC = roc_auc_score(y_true, y_prob)
    AUPRC = average_precision_score(y_true, y_prob)
    return {
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "Precision": PPV, "Recall": TPR, "Specificity": TNR,
        "Accuracy": Acc, "MCC": MCC, "F1": F1,
        "AUROC": AUROC, "AUPRC": AUPRC
    }

def pad_or_truncate(features_list, target_len):
    fixed = []
    for feat in features_list:
        arr = np.array(feat)
        if arr.ndim > 1:
            arr = arr.flatten()
        if len(arr) > target_len:
            fixed.append(arr[:target_len])
        elif len(arr) < target_len:
            pad = np.zeros(target_len - len(arr))
            fixed.append(np.concatenate([arr, pad]))
        else:
            fixed.append(arr)
    return np.vstack(fixed)

def load_fold_data(fold_num):
    fold_path = os.path.join(BASE_DIR, f"fold{fold_num}_pkl")
    with open(os.path.join(fold_path, f"positivedata_k{fold_num}.pkl"), "rb") as f:
        pos_train = pickle.load(f)
    with open(os.path.join(fold_path, f"positivedata_test_k{fold_num}.pkl"), "rb") as f:
        pos_test = pickle.load(f)
    with open(os.path.join(fold_path, f"negativedata_k{fold_num}.pkl"), "rb") as f:
        neg_train = pickle.load(f)
    with open(os.path.join(fold_path, f"negativedata_test_k{fold_num}.pkl"), "rb") as f:
        neg_test = pickle.load(f)
    return pos_train, pos_test, neg_train, neg_test

def load_all_train_data(target_len):
    all_pos, all_neg = [], []
    for fold_num in range(1, 6):
        pos_train, _, neg_train, _ = load_fold_data(fold_num)
        all_pos.extend(list(pos_train.values()))
        all_neg.extend(list(neg_train.values()))
    X = pad_or_truncate(all_pos + all_neg, target_len)
    y = np.array([1] * len(all_pos) + [0] * len(all_neg))
    return X, y

# ======================== 搜索空间与评估器 ========================
def build_search():
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),  # 稀疏/大维度更稳妥；不改变相对结构
        ("svc", svm.SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE))
    ])

    # 对数均匀分布抽样 C 与 gamma；class_weight 也一并探索
    param_distributions = {
        "svc__C": loguniform(1e-2, 1e3),
        "svc__gamma": loguniform(1e-4, 1),
        "svc__class_weight": [None, "balanced"],
    }

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_distributions,
        n_iter=24,                 # 可按算力调整
        scoring="average_precision",  # 以 AUPRC 为主目标
        n_jobs=-1,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE,
        verbose=1,
        refit=True                 # 以最优参数在该折训练集重拟合
    )
    return search

# ======================== 主流程 ========================
def main():
    per_fold_rows = []
    best_params_per_fold = []

    # ---------- 外层5折：每折上做“内层调参 + 外层测试评估” ----------
    for fold_num in range(1, 5 + 1):
        print(f"\n{'='*40}\n[Outer Fold {fold_num}] Inner-CV tuning & test eval")

        pos_train, pos_test, neg_train, neg_test = load_fold_data(fold_num)

        X_train_raw = list(pos_train.values()) + list(neg_train.values())
        y_train = np.array([1] * len(pos_train) + [0] * len(neg_train))
        X_test_raw = list(pos_test.values()) + list(neg_test.values())
        y_test = np.array([1] * len(pos_test) + [0] * len(neg_test))

        # 对齐长度（按该折 train+test 的最大长度）
        max_len = max(max(len(x) for x in X_train_raw), max(len(x) for x in X_test_raw))
        X_train = pad_or_truncate(X_train_raw, max_len)
        X_test = pad_or_truncate(X_test_raw, max_len)
        test_ids = list(pos_test.keys()) + list(neg_test.keys())

        # 内层调参
        search = build_search()
        search.fit(X_train, y_train)

        # 在该折测试集上评估
        y_prob = search.predict_proba(X_test)[:, 1]
        scores = multi_scores(y_test, y_prob, threshold=0.5)

        # 记录
        row = {"Fold": fold_num, **scores}
        row.update({f"param_{k}": v for k, v in search.best_params_.items()})
        row["best_cv_score(avg_prec)"] = float(search.best_score_)
        per_fold_rows.append(row)
        best_params_per_fold.append(search.best_params_)

        # 保存该折预测
        out_pred = os.path.join(RESULT_DIR, f"tune_fold{fold_num}_predictions.csv")
        pd.DataFrame({
            "Protein_ID": test_ids,
            "Label": y_test,
            "Pred_Prob": y_prob
        }).to_csv(out_pred, index=False)

        print(f"[Fold {fold_num}] AUPRC={scores['AUPRC']:.4f}, AUROC={scores['AUROC']:.4f}, "
              f"Recall={scores['Recall']:.4f}, F1={scores['F1']:.4f}")
        print(f"[Fold {fold_num}] Best params: {search.best_params_}")
        print(f"[Fold {fold_num}] Predictions saved to: {out_pred}")

    # 汇总外层评估
    df_folds = pd.DataFrame(per_fold_rows)
    df_folds.to_csv(os.path.join(RESULT_DIR, "tune_outerfold_scores.csv"), index=False)

    summary = df_folds[["AUPRC", "AUROC", "Recall", "F1", "MCC", "Accuracy", "Precision", "Specificity"]].agg(["mean", "std"])
    summary.to_csv(os.path.join(RESULT_DIR, "tune_outerfold_summary.csv"))

    # ---------- 选一个“代表性”的参数组合 ----------
    # 策略：以外层测试 AUPRC 均值最高的折的 best_params 作为候选；也可取各折 AUPRC 排名投票。
    best_idx = df_folds["AUPRC"].idxmax()
    rep_params = {
        k.replace("param_", ""): v
        for k, v in df_folds.loc[best_idx].items()
        if str(k).startswith("param_")
    }
    print("\n[Select Representative Params from Outer Test]")
    print(rep_params)

    # ---------- 全量训练集：再次大范围随机搜索，产出最终模型 ----------
    print(f"\n{'='*40}\n[Global Tuning on All Training Data]")
    # 先计算最终长度（按你原脚本的方法取所有训练的最大长度）
    final_len = 0
    for f in range(1, 6):
        pos_train, _, neg_train, _ = load_fold_data(f)
        lengths = [len(x) for x in list(pos_train.values()) + list(neg_train.values())]
        final_len = max(final_len, max(lengths))
    X_all, y_all = load_all_train_data(final_len)

    global_search = build_search()
    # 小技巧：以代表性参数作为“warm start”思路，缩小采样方差（通过设定 random_state 固定已足够）
    global_search.fit(X_all, y_all)
    print("[Global] best params:", global_search.best_params_)
    print("[Global] best cv avg_precision:", float(global_search.best_score_))

    # 用全量数据 + 最优参数训练最终模型并保存
    final_model = global_search.best_estimator_
    model_path = os.path.join(MODEL_DIR, "svm_final_tuned.joblib")
    dump(final_model, model_path)
    print(f"[Global] Final tuned model saved at: {model_path}")

    # 额外保存参数与汇总
    with open(os.path.join(RESULT_DIR, "tune_selected_params.txt"), "w") as f:
        f.write("Representative params from outer test:\n")
        f.write(str(rep_params) + "\n\n")
        f.write("Global best params via inner CV on all training data:\n")
        f.write(str(global_search.best_params_) + "\n")
        f.write(f"\nGlobal best cv (avg_precision): {float(global_search.best_score_):.6f}\n")

if __name__ == "__main__":
    main()
