
import os
import pickle
import numpy as np
from sklearn import svm
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef

def multi_scores(y_true, y_pred, show=False, threshold=0.5):
    y_bin = (y_pred >= threshold).astype(int)
    TP = ((y_true == 1) & (y_bin == 1)).sum()
    TN = ((y_true == 0) & (y_bin == 0)).sum()
    FP = ((y_true == 0) & (y_bin == 1)).sum()
    FN = ((y_true == 1) & (y_bin == 0)).sum()

    PPV = TP / (TP + FP + 1e-10)
    TPR = TP / (TP + FN + 1e-10)
    TNR = TN / (TN + FP + 1e-10)
    ACC = (TP + TN) / (TP + TN + FP + FN + 1e-10)
    F1 = 2 * PPV * TPR / (PPV + TPR + 1e-10)
    MCC = (TP * TN - FP * FN) / (np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)) + 1e-10)
    AUROC = roc_auc_score(y_true, y_pred)
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    AUPRC = auc(recall, precision)

    scores = [TP, TN, FP, FN, PPV, TPR, TNR, ACC, MCC, F1, AUROC, AUPRC]

    if show:
        print("TP: %d, TN: %d, FP: %d, FN: %d" % (TP, TN, FP, FN))
        print("PPV: %.4f, TPR: %.4f, TNR: %.4f, ACC: %.4f" % (PPV, TPR, TNR, ACC))
        print("MCC: %.4f, F1: %.4f, AUROC: %.4f, AUPRC: %.4f" % (MCC, F1, AUROC, AUPRC))

    return scores

output_dir = "./results/5folds_ESM2"
os.makedirs(output_dir, exist_ok=True)

test_scores = []

for foldn in range(1, 6):
    print(f"\n🟢 Fold {foldn} starting...")

    pos_train_path = f"esm2/fold{foldn}_pkl/positivedata_k{foldn}.pkl"
    neg_train_path = f"esm2/fold{foldn}_pkl/negativedata_k{foldn}.pkl"
    pos_test_path = f"esm2/fold{foldn}_pkl/positivedata_test_k{foldn}.pkl"
    neg_test_path = f"esm2/fold{foldn}_pkl/negativedata_test_k{foldn}.pkl"

    if not (os.path.exists(pos_train_path) and os.path.exists(neg_train_path) and os.path.exists(pos_test_path) and os.path.exists(neg_test_path)):
        print(f"❌ Missing files for fold {foldn}, skipping.")
        continue

    with open(pos_train_path, "rb") as f:
        pos_train = pickle.load(f)
    with open(neg_train_path, "rb") as f:
        neg_train = pickle.load(f)
    with open(pos_test_path, "rb") as f:
        pos_test = pickle.load(f)
    with open(neg_test_path, "rb") as f:
        neg_test = pickle.load(f)

    def extract_xy(d, label):
        X = list(d.values())
        y = [label] * len(X)
        return X, y

    X_pos_train, y_pos_train = extract_xy(pos_train, 1)
    X_neg_train, y_neg_train = extract_xy(neg_train, 0)
    X_pos_test, y_pos_test = extract_xy(pos_test, 1)
    X_neg_test, y_neg_test = extract_xy(neg_test, 0)

    X_train = np.array(X_pos_train + X_neg_train)
    y_train = np.array(y_pos_train + y_neg_train)
    X_test = np.array(X_pos_test + X_neg_test)
    y_test = np.array(y_pos_test + y_neg_test)

    model = svm.SVC(C=10, gamma=0.25, kernel='rbf', probability=True)
    model.fit(X_train, y_train)
    y_test_pred = model.predict_proba(X_test)[:, 1]

    test_score = multi_scores(y_test, y_test_pred, show=True)
    test_scores.append(test_score)

    # 保存预测结果
    with open(f"{output_dir}/test_pred_fold{foldn}.txt", "w") as f:
        for idx in range(len(y_test)):
            f.write(f"{y_test[idx]}\t{y_test_pred[idx]:.4f}\n")

    # 保存指标
    with open(f"{output_dir}/test_score_fold{foldn}.txt", "w") as f:
        f.write("TP\tTN\tFP\tFN\tPPV\tTPR\tTNR\tACC\tMCC\tF1\tAUROC\tAUPRC\n")
        f.write("\t".join([f"{x:.4f}" if isinstance(x, float) else str(x) for x in test_score]) + "\n")

    print(f"✅ Fold {foldn} done: Acc={test_score[7]:.4f}, AUROC={test_score[10]:.4f}")

# 计算平均值
test_scores = np.array(test_scores)
mean = test_scores.mean(axis=0)
std = test_scores.std(axis=0)
metric_names = ["TP", "TN", "FP", "FN", "PPV", "TPR", "TNR", "ACC", "MCC", "F1", "AUROC", "AUPRC"]

with open(f"{output_dir}/test_average_score.txt", "w") as f:
    f.write("Metric\tMean\tStd\n")
    for name, m, s in zip(metric_names, mean, std):
        f.write(f"{name}\t{m:.4f}\t{s:.4f}\n")

print("\n🎉 All folds completed. Summary:")
for name, m, s in zip(metric_names, mean, std):
    print(f"{name}: {m:.4f} ± {s:.4f}")
