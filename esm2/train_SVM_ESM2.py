import os
import pickle
import numpy as np
from sklearn import svm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score
)
import pandas as pd

# ✔️ 结果输出路径
result_dir = "esm2/esm2_svm_results"
os.makedirs(result_dir, exist_ok=True)

# ✔️ 需要的度量名称
metric_names = ["TP", "TN", "FP", "FN", "PPV", "TPR", "TNR", "Acc", "MCC", "F1", "AUROC", "AUPRC"]
test_scores = []

# ✔️ 文件是否存在的检查函数
def all_fold_files_exist(fold_path):
    required_files = [
        "positivedata_k1.pkl",
        "positivedata_test_k1.pkl",
        "negativedata_k1.pkl",
        "negativedata_test_k1.pkl"
    ]
    for f in required_files:
        if not os.path.isfile(os.path.join(fold_path, f)):
            print(f"  ⛔ Missing: {f}")
            return False
    return True

# ✔️ 开始 5-fold 训练与评估
for i in range(1, 6):
    print(f"\n🟢 Fold {i} starting...")
    fold_path = f"esm2/fold{i}_pkl"
    if not all_fold_files_exist(fold_path):
        print(f"❌ Missing files for fold {i}, skipping.\n")
        continue

    # ✔️ 加载数据
    
    with open(os.path.join(fold_path, f"positivedata_k{i}.pkl"), "rb") as f::
        pos_train = pickle.load(f)
    with open(os.path.join(fold_path, f"positivedata_test_k{i}.pkl"), "rb") as f:
        pos_test = pickle.load(f)
    with open(os.path.join(fold_path, f"negativedata_k{i}.pkl"), "rb") as f:
        neg_train = pickle.load(f)
    with open(os.path.join(fold_path, f1_score"negativedata_test_k{i}.pkl"), "rb") as f:
        neg_test = pickle.load(f)

    # ✔️ 构造训练集与测试集
    X_train = list(pos_train.values()) + list(neg_train.values())
    y_train = [1] * len(pos_train) + [0] * len(neg_train)
    X_test = list(pos_test.values()) + list(neg_test.values())
    y_test = [1] * len(pos_test) + [0] * len(neg_test)
    test_ids = list(pos_test.keys()) + list(neg_test.keys())

    # ✔️ 训练模型（可添加 class_weight="balanced"）
    clf = svm.SVC(kernel="rbf", C=10, gamma=0.25, probability=True)
    clf.fit(X_train, y_train)

    # ✔️ 预测与评估
    y_pred_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob >= 0.5).astype(int)

    TP = sum((y_test[i] == 1 and y_pred[i] == 1) for i in range(len(y_test)))
    TN = sum((y_test[i] == 0 and y_pred[i] == 0) for i in range(len(y_test)))
    FP = sum((y_test[i] == 0 and y_pred[i] == 1) for i in range(len(y_test)))
    FN = sum((y_test[i] == 1 and y_pred[i] == 0) for i in range(len(y_test)))

    PPV = precision_score(y_test, y_pred, zero_division=0)
    TPR = recall_score(y_test, y_pred, zero_division=0)
    TNR = TN / (TN + FP + 1e-6)
    ACC = accuracy_score(y_test, y_pred)
    MCC = np.corrcoef(y_test, y_pred)[0, 1] if TP + TN + FP + FN > 0 else 0
    F1 = f1_score(y_test, y_pred, zero_division=0)
    AUROC = roc_auc_score(y_test, y_pred_prob)
    AUPRC = average_precision_score(y_test, y_pred_prob)

    score = [TP, TN, FP, FN, PPV, TPR, TNR, ACC, MCC, F1, AUROC, AUPRC]
    test_scores.append(score)

    print(f"✅ Fold {i} done: Acc={ACC:.4f}, AUROC={AUROC:.4f}")

    # ✔️ 保存预测结果
    result_df = pd.DataFrame({
        "Protein_ID": test_ids,
        "Label": y_test,
        "Pred_Prob": y_pred_prob
    })
    result_df.to_csv(f"{result_dir}/fold{i}_predictions.csv", index=False)

# ✔️ 总结平均结果
test_scores = np.array(test_scores)
mean = test_scores.mean(axis=0)
std = test_scores.std(axis=0)

print("\n🎉 All folds completed. Summary:")
for name, m, s in zip(metric_names, mean, std):
    print(f"{name}: {m:.4f} ± {s:.4f}")

# ✔️ 保存总结果
df_mean = pd.DataFrame([mean], columns=metric_names)
df_std = pd.DataFrame([std], columns=metric_names)
df_mean.to_csv(os.path.join(result_dir, "5fold_mean.csv"), index=False)
df_std.to_csv(os.path.join(result_dir, "5fold_std.csv"), index=False)
