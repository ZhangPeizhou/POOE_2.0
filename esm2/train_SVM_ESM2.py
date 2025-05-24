import os
import pickle
import numpy as np
from sklearn import svm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score
)
import pandas as pd

# ======================== 全局配置 ========================
BASE_DIR = "/content/POOE_2.0"  # Colab根目录
RESULT_DIR = os.path.join(BASE_DIR, "esm2/esm2_svm_results")
os.makedirs(RESULT_DIR, exist_ok=True)

# ======================== 评估函数 ========================
def multi_scores(y_true, y_prob, threshold=0.5):
    """与trainning_SVM完全一致的评估逻辑"""
    y_pred = (y_prob >= threshold).astype(int)
    
    # 混淆矩阵指标
    TN, FP, FN, TP = confusion_matrix(y_true, y_pred).ravel()
    
    # 计算核心指标
    PPV = precision_score(y_true, y_pred, zero_division=0)
    TPR = recall_score(y_true, y_pred, zero_division=0)
    TNR = TN / (TN + FP) if (TN + FP) > 0 else 0
    Acc = accuracy_score(y_true, y_pred)
    F1 = f1_score(y_true, y_pred, zero_division=0)
    
    # MCC计算（修正后的公式）
    denominator = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    MCC = (TP * TN - FP * FN) / denominator if denominator != 0 else 0
    
    # AUC相关
    AUROC = roc_auc_score(y_true, y_prob)
    AUPRC = average_precision_score(y_true, y_prob)
    
    return [TP, TN, FP, FN, PPV, TPR, TNR, Acc, MCC, F1, AUROC, AUPRC]

# ======================== 文件检查 ========================
def validate_fold_files(fold_path, fold_num):
    """检查当前fold目录下是否存在必需文件"""
    required_files = [
        f"positivedata_k{fold_num}.pkl",
        f"positivedata_test_k{fold_num}.pkl",
        f"negativedata_k{fold_num}.pkl",
        f"negativedata_test_k{fold_num}.pkl"
    ]
    
    missing_files = []
    for f in required_files:
        if not os.path.isfile(os.path.join(fold_path, f)):
            missing_files.append(f)
    
    if missing_files:
        print(f"  ⛔ Fold {fold_num}缺失文件:")
        for f in missing_files:
            print(f"    - {os.path.join(fold_path, f)}")
        return False
    return True

# ======================== 主逻辑 ========================
def main():
    test_scores = []
    
    for fold_num in range(1, 6):  # 遍历5个fold
        print(f"\n{'='*40}\n🟢 Processing Fold {fold_num}")
        
        # 构建绝对路径
        fold_path = os.path.join(BASE_DIR, "esm2", f"fold{fold_num}_pkl")
        print(f"   Path: {fold_path}")
        
        # 文件存在性检查
        if not validate_fold_files(fold_path, fold_num):
            print(f"❌ Skipping Fold {fold_num}\n")
            continue
            
        # 加载数据 -------------------------------------------------
        print("   Loading data...")
        try:
            with open(os.path.join(fold_path, f"positivedata_k{fold_num}.pkl"), "rb") as f:
                pos_train = pickle.load(f)
            with open(os.path.join(fold_path, f"positivedata_test_k{fold_num}.pkl"), "rb") as f:
                pos_test = pickle.load(f)
            with open(os.path.join(fold_path, f"negativedata_k{fold_num}.pkl"), "rb") as f:
                neg_train = pickle.load(f)
            with open(os.path.join(fold_path, f"negativedata_test_k{fold_num}.pkl"), "rb") as f:
                neg_test = pickle.load(f)
        except Exception as e:
            print(f"   🔴 数据加载失败: {str(e)}")
            continue
            
        # 构建数据集 -----------------------------------------------
        X_train = list(pos_train.values()) + list(neg_train.values())
        y_train = [1] * len(pos_train) + [0] * len(neg_train)
        X_test = list(pos_test.values()) + list(neg_test.values())
        y_test = [1] * len(pos_test) + [0] * len(neg_test)
        test_ids = list(pos_test.keys()) + list(neg_test.keys())
        
        # 训练模型 -----------------------------------------------
        print("   Training SVM...")
        clf = svm.SVC(kernel="rbf", C=10, gamma=0.25, probability=True)
        clf.fit(X_train, y_train)
        
        # 预测与评估 ---------------------------------------------
        print("   Evaluating...")
        y_prob = clf.predict_proba(X_test)[:, 1]
        fold_scores = multi_scores(y_test, y_prob)
        test_scores.append(fold_scores)
        
        # 保存预测结果 -------------------------------------------
        output_file = os.path.join(RESULT_DIR, f"test_pred_{fold_num-1}.txt")
        with open(output_file, "w") as f:
            for protein_id, true_label, prob in zip(test_ids, y_test, y_prob):
                f.write(f"{protein_id}\t{true_label}\t{prob:.6f}\n")
                
        # 保存指标结果 -------------------------------------------
        score_file = os.path.join(RESULT_DIR, f"test_score_{fold_num-1}.txt")
        with open(score_file, "w") as f:
            f.write("TP\tTN\tFP\tFN\tPPV\tTPR\tTNR\tAcc\tmcc\tf1\tAUROC\tAUPRC\n")
            f.write("\t".join([f"{x:.4f}" for x in fold_scores]))
        
        print(f"   ✅ Fold {fold_num}完成: Acc={fold_scores[7]:.4f}, AUROC={fold_scores[10]:.4f}")

    # 汇总结果 -------------------------------------------------
    if test_scores:
        print(f"\n{'='*40}\n🎉 5-Fold汇总结果:")
        test_scores = np.array(test_scores)
        fmat = [1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3]  # 小数位数控制
        
        with open(os.path.join(RESULT_DIR, "test_average_score.txt"), "w") as f:
            header = "TP\tTN\tFP\tFN\tPPV\tTPR\tTNR\tAcc\tmcc\tf1\tAUROC\tAUPRC\n"
            f.write(header)
            
            mean_scores = test_scores.mean(axis=0)
            std_scores = test_scores.std(axis=0)
            
            line = "\t".join([
                f"{mean:.{decimals}f}±{std:.{decimals}f}"
                for mean, std, decimals in zip(mean_scores, std_scores, fmat)
            ])
            f.write(line)
            
            # 控制台输出
            print(header.strip())
            print("\t".join([f"{x:.4f}" for x in mean_scores]))
    else:
        print("\n⚠️ 无有效结果可汇总")

if __name__ == "__main__":
    main()