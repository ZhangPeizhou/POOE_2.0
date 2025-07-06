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
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
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

# ======================== 数据加载函数 ========================
def load_fold_data(fold_num):
    fold_dir = os.path.join(BASE_DIR, f"fold{fold_num}_pkl")
    files = {
        "pos_train": f"positivedata_k{fold_num}.pkl",
        "neg_train": f"negativedata_k{fold_num}.pkl",
        "pos_test": f"positivedata_test_k{fold_num}.pkl",
        "neg_test": f"negativedata_test_k{fold_num}.pkl"
    }
    for f in files.values():
        if not os.path.isfile(os.path.join(fold_dir, f)):
            raise FileNotFoundError(f"missing file: {f}")
    with open(os.path.join(fold_dir, files["pos_train"]), "rb") as f:
        pos_train = pickle.load(f)
    with open(os.path.join(fold_dir, files["neg_train"]), "rb") as f:
        neg_train = pickle.load(f)
    with open(os.path.join(fold_dir, files["pos_test"]), "rb") as f:
        pos_test = pickle.load(f)
    with open(os.path.join(fold_dir, files["neg_test"]), "rb") as f:
        neg_test = pickle.load(f)
    return pos_train, pos_test, neg_train, neg_test

def load_all_train_data():
    all_pos, all_neg = [], []
    for fold_num in range(1, 6):
        pos_train, _, neg_train, _ = load_fold_data(fold_num)
        all_pos.extend(list(pos_train.values()))
        all_neg.extend(list(neg_train.values()))
    X = all_pos + all_neg
    y = [1] * len(all_pos) + [0] * len(all_neg)
    return X, y

# ======================== 主逻辑 ========================
def main():
    test_scores = []

    # Phase 1: 5-fold交叉验证
    for fold_num in range(1, 6):
        try:
            print(f"\n{'='*40}\n Cross-Validation Fold {fold_num}")
            fold_dir = os.path.join(BASE_DIR, f"fold{fold_num}_pkl")
            result_dir = os.path.join(fold_dir, "results")
            os.makedirs(result_dir, exist_ok=True)

            pos_train, pos_test, neg_train, neg_test = load_fold_data(fold_num)

            # 构建数据集
            X_train = list(pos_train.values()) + list(neg_train.values())
            y_train = [1] * len(pos_train) + [0] * len(neg_train)
            X_test = list(pos_test.values()) + list(neg_test.values())
            y_test = [1] * len(pos_test) + [0] * len(neg_test)
            test_ids = list(pos_test.keys()) + list(neg_test.keys())

            # 训练模型
            clf = svm.SVC(kernel="rbf", C=10, gamma=0.25, probability=True)
            clf.fit(X_train, y_train)

            # 评估
            y_prob = clf.predict_proba(X_test)[:, 1]
            fold_scores = multi_scores(y_test, y_prob)
            test_scores.append(fold_scores)

            # 保存预测结果
            pred_file = os.path.join(result_dir, f"fold{fold_num}_predictions.csv")
            pd.DataFrame({
                "Protein_ID": test_ids,
                "Label": y_test,
                "Pred_Prob": y_prob
            }).to_csv(pred_file, index=False)

            # 保存指标
            score_file = os.path.join(result_dir, f"fold{fold_num}_scores.txt")
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

            print(f" Fold {fold_num} results saved in {result_dir}")

        except Exception as e:
            print(f"Fold {fold_num} failed: {e}")
            continue

    # Phase 2: 训练最终模型
    try:
        print("\n" + "="*40 + "\n Training Final Model")
        X_all, y_all = load_all_train_data()
        final_model = svm.SVC(kernel="rbf", C=10, gamma=0.25, probability=True)
        final_model.fit(X_all, y_all)
        model_path = os.path.join(MODEL_DIR, "svm_final.joblib")
        dump(final_model, model_path)
        print(f"Final model saved at: {model_path}")
    except Exception as e:
        print(f"Final model training failed: {str(e)}")

    # Phase 3: 汇总评估结果
    if test_scores:
        test_scores = np.array(test_scores)
        print("\n" + "="*40 + "\n 5-Fold Summary:")
        print("{:<12} | {:^10} | {:^10}".format("Metric", "Mean", "Std"))
        print("-" * 36)
        metrics = [
            ("Precision", 4, 3), ("Recall", 5, 3), ("Specificity", 6, 3),
            ("Accuracy", 7, 3), ("Mcc", 8, 3), ("F1", 9, 3),
            ("AUROC", 10, 3), ("AUPRC", 11, 3)
        ]
        summary_file = os.path.join(BASE_DIR, "cross_validation_summary.txt")
        with open(summary_file, "w") as f:
            f.write("Metric\tMean\tStd\n")
            for name, idx, dec in metrics:
                mean = test_scores[:, idx].mean()
                std = test_scores[:, idx].std()
                print(f"{name:<12} | {mean:.{dec}f} ± {std:.{dec}f}")
                f.write(f"{name}\t{mean:.{dec}f}\t{std:.{dec}f}\n")
        print(f"\nCross-validation summary saved at: {summary_file}")

if __name__ == "__main__":
    main()
