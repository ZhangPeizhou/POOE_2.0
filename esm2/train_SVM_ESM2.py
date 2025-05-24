import os
import pickle
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.calibration import CalibratedClassifierCV

# 输出目录
output_dir = "esm2_svm_results"
os.makedirs(output_dir, exist_ok=True)

# 记录五折评估结果
fold_results = []

def load_pkl_features(pos_pkl, neg_pkl, label_pos=1, label_neg=0):
    pos = pickle.load(open(pos_pkl, "rb"))
    neg = pickle.load(open(neg_pkl, "rb"))
    X = list(pos.values()) + list(neg.values())
    y = [label_pos] * len(pos) + [label_neg] * len(neg)
    names = list(pos.keys()) + list(neg.keys())
    return np.array(X), np.array(y), names

for k in range(1, 6):
    print(f"\n🟢 Fold {k} starting...")

    base_path = f"./fold{k}_pkl"
    pos_train = f"{base_path}/positivedata_k{k}.pkl"
    pos_test = f"{base_path}/positivedata_test_k{k}.pkl"
    neg_train = f"{base_path}/negativedata_k{k}.pkl"
    neg_test = f"{base_path}/negativedata_test_k{k}.pkl"

    if not all(os.path.exists(p) for p in [pos_train, pos_test, neg_train, neg_test]):
        print(f"❌ Missing files for fold {k}, skipping.")
        continue

    X_train, y_train, _ = load_pkl_features(pos_train, neg_train)
    X_test, y_test, test_ids = load_pkl_features(pos_test, neg_test)

    # 模型：标准化 + SVM + 概率输出
    model = make_pipeline(
        StandardScaler(),
        CalibratedClassifierCV(svm.SVC(C=10, gamma=0.25, kernel='rbf'), cv=3)
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # 评估指标
    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_prob)
    auc_pr = average_precision_score(y_test, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn) + 1e-6)

    fold_results.append([tp, tn, fp, fn, pre, rec, tn / (tn + fp), acc, mcc, f1, auc_roc, auc_pr])
    print(f"✅ Fold {k} done: Acc={acc:.4f}, AUROC={auc_roc:.4f}")

    # 保存预测分数
    df = pd.DataFrame({
        "ProteinID": test_ids,
        "TrueLabel": y_test,
        "PredictedProb": y_prob
    })
    df.to_csv(f"{output_dir}/fold{k}_predictions.csv", index=False)

# 平均结果输出
metrics_names = ["TP", "TN", "FP", "FN", "PPV", "TPR", "TNR", "Acc", "MCC", "F1", "AUROC", "AUPRC"]
fold_results = np.array(fold_results)
mean_result = fold_results.mean(axis=0)
std_result = fold_results.std(axis=0)

# 保存评估结果
df_mean = pd.DataFrame([mean_result], columns=metrics_names)
df_std = pd.DataFrame([std_result], columns=metrics_names)
df_mean.to_csv(f"{output_dir}/final_scores_mean.csv", index=False)
df_std.to_csv(f"{output_dir}/final_scores_std.csv", index=False)

print("\n🎉 All folds completed. Summary:")
for name, mean, std in zip(metrics_names, mean_result, std_result):
    print(f"{name}: {mean:.4f} ± {std:.4f}")
