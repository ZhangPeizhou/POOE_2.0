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

# ======================== 数据加载 ========================
def load_fold_data(fold_num):
    """加载单个fold的数据（修复版）"""
    fold_path = os.path.join(BASE_DIR, "esm2", f"fold{fold_num}_pkl")
    
    # 检查文件是否存在
    required_files = [
        f"positivedata_k{fold_num}.pkl",
        f"positivedata_test_k{fold_num}.pkl",
        f"negativedata_k{fold_num}.pkl",
        f"negativedata_test_k{fold_num}.pkl"
    ]
    for f in required_files:
        if not os.path.isfile(os.path.join(fold_path, f)):
            raise FileNotFoundError(f"⛔ 文件缺失: {os.path.join(fold_path, f)}")
    
    # 加载数据
    with open(os.path.join(fold_path, f"positivedata_k{fold_num}.pkl"), "rb") as f:
        pos_train = pickle.load(f)
    with open(os.path.join(fold_path, f"positivedata_test_k{fold_num}.pkl"), "rb") as f:
        pos_test = pickle.load(f)
    with open(os.path.join(fold_path, f"negativedata_k{fold_num}.pkl"), "rb") as f:
        neg_train = pickle.load(f)
    with open(os.path.join(fold_path, f"negativedata_test_k{fold_num}.pkl"), "rb") as f:
        neg_test = pickle.load(f)
    
    return pos_train, pos_test, neg_train, neg_test

def load_all_train_data():
    """合并所有fold的训练数据（最终模型用）"""
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
            print(f"\n{'='*40}\n🟢 Cross-Validation Fold {fold_num}")
            pos_train, pos_test, neg_train, neg_test = load_fold_data(fold_num)
            
            # 构建数据集
            X_train = list(pos_train.values()) + list(neg_train.values())
            y_train = [1] * len(pos_train) + [0] * len(neg_train)
            X_test = list(pos_test.values()) + list(neg_test.values())
            y_test = [1] * len(pos_test) + [0] * len(neg_test)
            
            # 训练模型
            clf = svm.SVC(kernel="rbf", C=10, gamma=0.25, probability=True)
            clf.fit(X_train, y_train)
            
            # 评估
            y_prob = clf.predict_proba(X_test)[:, 1]
            fold_scores = multi_scores(y_test, y_prob)
            test_scores.append(fold_scores)
            print(f"✅ Fold {fold_num} 评估完成")
        except Exception as e:
            print(f"🔴 Fold {fold_num} 失败: {str(e)}")
            continue
    
    # Phase 2: 训练最终模型
    try:
        print("\n{'='*40}\n🟢 Training Final Model")
        X_all, y_all = load_all_train_data()
        final_model = svm.SVC(kernel="rbf", C=10, gamma=0.25, probability=True)
        final_model.fit(X_all, y_all)
        dump(final_model, os.path.join(MODEL_DIR, "svm_final.joblib"))
        print(f"✅ 最终模型保存至: {MODEL_DIR}/svm_final.joblib")
    except Exception as e:
        print(f"🔴 最终模型训练失败: {str(e)}")
    
    # Phase 3: 输出结果
    if test_scores:
        test_scores = np.array(test_scores)
        print(f"\n{'='*40}\n📊 5-Fold 平均性能")
        print("Metric     | Mean ± Std")
        print("-----------------------")
        metrics = [
            ("PPV", 4, 4), ("TPR", 5, 4), ("TNR", 6, 4),
            ("Acc", 7, 4), ("AUROC", 10, 4), ("AUPRC", 11, 4)
        ]
        for name, idx, dec in metrics:
            mean = test_scores[:, idx].mean()
            std = test_scores[:, idx].std()
            print(f"{name:<8} | {mean:.{dec}f} ± {std:.{dec}f}")

if __name__ == "__main__":
    main()
