import os
import pickle
import numpy as np
from sklearn import svm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score
)
import pandas as pd

# ======================== 新增函数 ========================
def multi_scores(y_true, y_prob, threshold=0.5, show=False):
    y_pred = (y_prob >= threshold).astype(int)
    
    # 基础指标
    TN, FP, FN, TP = confusion_matrix(y_true, y_pred).ravel()
    PPV = precision_score(y_true, y_pred, zero_division=0)
    TPR = recall_score(y_true, y_pred, zero_division=0)
    TNR = TN / (TN + FP) if (TN + FP) > 0 else 0
    Acc = accuracy_score(y_true, y_pred)
    F1 = f1_score(y_true, y_pred, zero_division=0)
    
    # MCC 计算
    denominator = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    MCC = (TP * TN - FP * FN) / denominator if denominator != 0 else 0
    
    # AUC 相关
    AUROC = roc_auc_score(y_true, y_prob)
    AUPRC = average_precision_score(y_true, y_prob)
    
    return [TP, TN, FP, FN, PPV, TPR, TNR, Acc, MCC, F1, AUROC, AUPRC]

# ======================== 修改后的主逻辑 ========================
result_dir = "esm2/esm2_svm_results"
os.makedirs(result_dir, exist_ok=True)

# ✔️ 文件检查逻辑保持不变
def all_fold_files_exist(fold_path):
    required_files = [f"positivedata_k{i}.pkl" for i in range(1,6)] + \
                    [f"negativedata_k{i}.pkl" for i in range(1,6)]
    for f in required_files:
        if not os.path.isfile(os.path.join(fold_path, f)):
            return False
    return True

test_scores = []

for i in range(1, 6):
    print(f"\n🟢 Fold {i} starting...")
    fold_path = f"esm2/fold{i}_pkl"
    if not all_fold_files_exist(fold_path):
        print(f"❌ Missing files for fold {i}, skipping.\n")
        continue

    # ✔️ 数据加载保持不变
    with open(os.path.join(fold_path, f"positivedata_k{i}.pkl"), "rb") as f:
        pos_train = pickle.load(f)
    with open(os.path.join(fold_path, f"positivedata_test_k{i}.pkl"), "rb") as f:
        pos_test = pickle.load(f)
    with open(os.path.join(fold_path, f"negativedata_k{i}.pkl"), "rb") as f:
        neg_train = pickle.load(f)
    with open(os.path.join(fold_path, f"negativedata_test_k{i}.pkl"), "rb") as f:
        neg_test = pickle.load(f)

    # ✔️ 数据集构造保持不变
    X_train = list(pos_train.values()) + list(neg_train.values())
    y_train = [1] * len(pos_train) + [0] * len(neg_train)
    X_test = list(pos_test.values()) + list(neg_test.values())
    y_test = [1] * len(pos_test) + [0] * len(neg_test)
    test_ids = list(pos_test.keys()) + list(neg_test.keys())

    # ✔️ 训练逻辑保持不变
    clf = svm.SVC(kernel="rbf", C=10, gamma=0.25, probability=True)
    clf.fit(X_train, y_train)
    
    # ======================== 新评估方式 ========================
    y_prob = clf.predict_proba(X_test)[:, 1]
    test_score = multi_scores(y_test, y_prob, show=True)
    test_scores.append(test_score)
    
    # ======================== 新输出格式 ========================
    # 保存预测结果（改用制表符分隔）
    with open(f"{result_dir}/test_pred_{i-1}.txt", "w") as f:
        for idx, (true, prob) in enumerate(zip(y_test, y_prob)):
            line = f"{test_ids[idx]}\t{true}\t{prob:.6f}\n"
            f.write(line)
    
    # 保存指标结果（与trainning_SVM一致）
    with open(f"{result_dir}/test_score_{i-1}.txt", "w") as f:
        f.write("TP\tTN\tFP\tFN\tPPV\tTPR\tTNR\tAcc\tmcc\tf1\tAUROC\tAUPRC\n")
        f.write("\t".join([f"{x:.4f}" for x in test_score]))

# ======================== 最终汇总输出 ========================
print("\n5 fold average")
test_scores = np.array(test_scores)
fmat =  [1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3]  # 控制小数位数

with open(f"{result_dir}/test_average_score.txt", "w") as f:
    line1 = "TP\tTN\tFP\tFN\tPPV\tTPR\tTNR\tAcc\tmcc\tf1\tAUROC\tAUPRC\n"
    line2 = "\t".join([f"{a:.{_}f}±{b:.{_}f}" 
                      for (_, a, b) in zip(fmat, test_scores.mean(0), test_scores.std(0))])
    f.write(line1)
    f.write(line2)

print("-----------------------------------")