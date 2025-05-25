# ======================== 主逻辑修改部分 ========================
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
            
            # ======================== 新增：保存当前fold评估结果 ========================
            score_file = os.path.join(RESULT_DIR, f"fold{fold_num}_scores.txt")
            with open(score_file, "w") as f:
                f.write("Metric          | Value\n")
                f.write("----------------|---------\n")
                metrics = [
                    ("TP", 0, 0), ("TN", 1, 0), ("FP", 2, 0), ("FN", 3, 0),
                    ("PPV", 4, 4), ("TPR", 5, 4), ("TNR", 6, 4),
                    ("Acc", 7, 4), ("mcc", 8, 4), ("F1", 9, 4),
                    ("AUROC", 10, 4), ("AUPRC", 11, 4)
                ]
                for name, idx, dec in metrics:
                    value = fold_scores[idx]
                    f.write(f"{name:<15} | {value:.{dec}f}\n")
            print(f"✅ Fold {fold_num} 评估结果保存至: {score_file}")
            
        except Exception as e:
            print(f"🔴 Fold {fold_num} 失败: {str(e)}")
            continue

    # ... (后续保持原有最终模型训练和汇总逻辑不变)
