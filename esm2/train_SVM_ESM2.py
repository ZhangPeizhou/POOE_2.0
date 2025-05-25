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
BASE_DIR = "/content/POOE_2.0"
RESULT_DIR = os.path.join(BASE_DIR, "esm2/esm2_svm_results")
MODEL_DIR = os.path.join(BASE_DIR, "esm2/saved_models")
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

# ======================== 数据加载与验证 ========================
def load_fold_data(fold_num):
    """加载单个fold的数据"""
    fold_path = os.path.join(BASE_DIR, "esm2", f"fold{fold_num}_pkl")
    required_files = [
        f"positivedata_k{fold_num}.pkl", f"positivedata_test_k{fold_num}.pkl",
        f"negativedata_k{fold_num}.pkl", f"negativedata_test_k{fold_num}.pkl"
    ]
    # 文件存在性检查（略，同之前逻辑）
    # 加载数据（略，同之前逻辑）
    return (pos_train, pos_test, neg_train, neg_test)

def load_all_train_data():
    """合并所有fold的训练数据（用于最终模型）"""
    all_pos, all_neg = [], []
    for fold_num in range(1, 6):
        pos_train, _, neg_train, _ = load_fold_data(fold_num)
        all_pos.extend(list(pos_train.values()))
        all_neg.extend(list(neg_train.values()))
    X = all_pos + all_neg
    y = [1]*len(all_pos) + [0]*len(all_neg)
    return X, y

# ======================== 主逻辑 ========================
def main():
    test_scores = []
    
    # Phase 1: 5-fold交叉验证评估 ----------------------------
    for fold_num in range(1, 6):
        print(f"\n{'='*40}\n🟢 Cross-Validation Fold {fold_num}")
        
        # 加载当前fold数据
        pos_train, pos_test, neg_train, neg_test = load_fold_data(fold_num)
        
        # 构建训练集/测试集
        X_train = list(pos_train.values()) + list(neg_train.values())
        y_train = [1]*len(pos_train) + [0]*len(neg_train)
        X_test = list(pos_test.values()) + list(neg_test.values())
        y_test = [1]*len(pos_test) + [0]*len(neg_test)
        
        # 训练临时模型（仅用于评估，不保存）
        clf = svm.SVC(kernel="rbf", C=10, gamma=0.25, probability=True)
        clf.fit(X_train, y_train)
        
        # 评估
        y_prob = clf.predict_proba(X_test)[:, 1]
        fold_scores = multi_scores(y_test, y_prob)
        test_scores.append(fold_scores)
        
        # 保存当前fold的评估结果
        result_df = pd.DataFrame({
            "Protein_ID": list(pos_test.keys()) + list(neg_test.keys()),
            "Label": y_test,
            "Pred_Prob": y_prob
        })
        result_df.to_csv(f"{RESULT_DIR}/fold{fold_num}_predictions.csv", index=False)
        print(f"✅ Fold {fold_num}评估完成")

    # Phase 2: 全体数据训练最终模型 ----------------------------
    print("\n{'='*40}\n🟢 Training Final Model on All Data")
    X_all, y_all = load_all_train_data()
    
    final_model = svm.SVC(kernel="rbf", C=10, gamma=0.25, probability=True)
    final_model.fit(X_all, y_all)
    
    # 保存最终模型
    model_path = os.path.join(MODEL_DIR, "svm_final_model.joblib")
    dump(final_model, model_path)
    print(f"✅ 最终模型已保存至: {model_path}")

    # Phase 3: 输出交叉验证汇总结果 ----------------------------
    print("\n{'='*40}\n📊 5-Fold交叉验证汇总")
    test_scores = np.array(test_scores)
    metrics = [
        ("TP", 0, 0), ("TN", 1, 0), ("FP", 2, 0), ("FN", 3, 0),
        ("PPV", 4, 4), ("TPR", 5, 4), ("TNR", 6, 4), ("Acc", 7, 4),
        ("mcc", 8, 4), ("F1", 9, 4), ("AUROC", 10, 4), ("AUPRC", 11, 4)
    ]
    
    # 控制台输出
    print("\n{:<8} | {:^12} | {:^12}".format("Metric", "Mean", "Std"))
    print("-"*38)
    for name, idx, dec in metrics:
        mean = test_scores[:, idx].mean()
        std = test_scores[:, idx].std()
        print(f"{name:<8} | {mean:.{dec}f} ± {std:.{dec}f}")

    # 文件输出
    with open(f"{RESULT_DIR}/cross_validation_summary.txt", "w") as f:
        f.write("Metric\tMean\tStd\n")
        for name, idx, dec in metrics:
            mean = test_scores[:, idx].mean()
            std = test_scores[:, idx].std()
            f.write(f"{name}\t{mean:.{dec}f}\t{std:.{dec}f}\n")

if __name__ == "__main__":
    main()
