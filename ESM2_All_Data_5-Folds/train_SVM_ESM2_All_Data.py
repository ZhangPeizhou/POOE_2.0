import os
import pickle
import numpy as np
from sklearn import svm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score
)
from joblib import dump
import pandas as pd

# ======================== 全局配置 ========================
BASE_DIR = "/content/POOE_2.0/ESM2_All_Data_5-Folds"
RESULT_DIR = os.path.join(BASE_DIR, "results")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ======================== 评估函数 ========================
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
    return [TP, TN, FP, FN, PPV, TPR, TNR, Acc, MCC, F1, AUROC, AUPRC]

# ======================== 长度统一函数 ========================
def pad_or_truncate(features_list, target_len):
    fixed = []
    for feat in features_list:
        feat = np.array(feat)
        if feat.ndim > 1:
            feat = feat.flatten()
        if len(feat) > target_len:
            fixed.append(feat[:target_len])
        elif len(feat) < target_len:
            pad = np.zeros(target_len - len(feat))
            fixed.append(np.concatenate([feat, pad]))
        else:
            fixed.append(feat)
    return fixed

# ======================== 数据加载 ========================
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
    y = [1] * len(all_pos) + [0] * len(all_neg)
    return X, y

# ======================== 主逻辑 ========================
def main():
    test_scores = []

    # Phase 1: 交叉验证
    for fold_num in range(1, 6):
        try:
            print(f"\n{'='*40}\n Cross-Validation Fold {fold_num}")
            pos_train, pos_test, neg_train, neg_test = load_fold_data(fold_num)

            # 构建数据集
            X_train_raw = list(pos_train.values()) + list(neg_train.values())
            y_train = [1] * len(pos_train) + [0] * len(neg_train)
            X_test_raw = list(pos_test.values()) + list(neg_test.values())
            y_test = [1] * len(pos_test) + [0] * len(neg_test)
            test_ids = list(pos_test.keys()) + list(neg_test.keys())

            # 统一特征长度（用训练集和测试集的最大长度）
            max_len = max(max(len(x) for x in X_train_raw), max(len(x) for x in X_test_raw))
            X_train = pad_or_truncate(X_train_raw, max_len)
            X_test = pad_or_truncate(X_test_raw, max_len)

            # 训练 SVM
            clf = svm.SVC(kernel="rbf", C=10, gamma=0.25, probability=True)
            clf.fit(X_train, y_train)

            # 评估
            y_prob = clf.predict_proba(X_test)[:, 1]
            fold_scores = multi_scores(y_test, y_prob)
            test_scores.append(fold_scores)

            # 保存预测结果
            pred_file = os.path.join(RESULT_DIR, f"fold{fold_num}_predictions.csv")
            pd.DataFrame({"Protein_ID": test_ids, "Label": y_test, "Pred_Prob": y_prob}).to_csv(pred_file, index=False)

            # 保存评估指标
            score_file = os.path.join(RESULT_DIR, f"fold{fold_num}_scores.txt")
            with open(score_file, "w") as f:
                f.write("Metric          | Value\n")
                f.write("----------------|---------\n")
                metrics = [
                    ("TP", 0, 0), ("TN", 1, 0), ("FP", 2, 0), ("FN", 3, 0),
                    ("Precision", 4, 3), ("Recall", 5, 3), ("Specificity", 6, 3),
                    ("Accuracy", 7, 3), ("Mcc", 8, 3), ("F1", 9, 3),
                    ("AUROC", 10, 3), ("AUPRC", 11, 3)
                ]
                for name, idx, dec in metrics:
                    f.write(f"{name:<15} | {fold_scores[idx]:.{dec}f}\n")

            print(f" Fold {fold_num} result saved at: {pred_file}")

        except Exception as e:
            print(f" Fold {fold_num} failed: {str(e)}")

    # Phase 2: 训练最终模型
    try:
        print(f"\n{'='*40}\n Training Final Model")
        final_len = max(
            max(len(x) for x in list(load_fold_data(f)[0].values()) + list(load_fold_data(f)[2].values()))
            for f in range(1, 6)
        )
        X_all, y_all = load_all_train_data(final_len)
        final_model = svm.SVC(kernel="rbf", C=10, gamma=0.25, probability=True)
        final_model.fit(X_all, y_all)
        model_path = os.path.join(MODEL_DIR, "svm_final.joblib")
        dump(final_model, model_path)
        print(f" Final model saved at: {model_path}")
    except Exception as e:
        print(f" Final model training failed: {str(e)}")

    # Phase 3: 汇总结果
    if test_scores:
        test_scores = np.array(test_scores)
        print(f"\n{'='*40}\n 5-Fold Summary")
        print("{:<8} | {:^10} | {:^10}".format("Metric", "Mean", "Std"))
        print("----------------------------------")
        metrics = [
            ("Precision", 4, 3), ("Recall", 5, 3), ("Specificity", 6, 3),
            ("Accuracy", 7, 3), ("Mcc", 8, 3), ("F1", 9, 3),
            ("AUROC", 10, 3), ("AUPRC", 11, 3)
        ]
        with open(os.path.join(RESULT_DIR, "cross_validation_summary.txt"), "w") as f:
            f.write("Metric\tMean\tStd\n")
            for name, idx, dec in metrics:
                mean = test_scores[:, idx].mean()
                std = test_scores[:, idx].std()
                print(f"{name:<8} | {mean:.{dec}f} ± {std:.{dec}f}")
                f.write(f"{name}\t{mean:.{dec}f}\t{std:.{dec}f}\n")

if __name__ == "__main__":
    main()