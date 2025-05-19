import os
import pickle

def load_seqnames(txt_file):
    with open(txt_file) as f:
        return [line.strip().replace(">", "") for line in f]

def extract_subset(all_data, names):
    missing = [n for n in names if n not in all_data]
    if missing:
        print(f"⚠️ {len(missing)} not found in .pkl: {missing[:3]}")
    return [all_data[n] for n in names if n in all_data]

def build_fold_pkl(fold_id, pos_pkl_path, neg_pkl_path, split_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    with open(pos_pkl_path, "rb") as f:
        pos_all = pickle.load(f)
    with open(neg_pkl_path, "rb") as f:
        neg_all = pickle.load(f)

    # ⚠️ 正样本不做划分，统一使用
    pos_train_ids = list(pos_all.keys())
    pos_test_ids = list(pos_all.keys())

    # ⚠️ 负样本用作者提供的划分
    neg_train_ids = load_seqnames(os.path.join(split_dir, f"x_train1670_seqname_k{fold_id}.txt"))
    neg_test_ids  = load_seqnames(os.path.join(split_dir, f"x_test1670_seqname_k{fold_id}.txt"))

    print(f"[Fold {fold_id}] Pos train: {len(pos_train_ids)}, test: {len(pos_test_ids)}")
    print(f"[Fold {fold_id}] Neg train: {len(neg_train_ids)}, test: {len(neg_test_ids)}")

    pickle.dump(extract_subset(pos_all, pos_train_ids), open(f"{out_dir}/positivedata_k{fold_id}.pkl", "wb"))
    pickle.dump(extract_subset(pos_all, pos_test_ids), open(f"{out_dir}/positivedata_test_k{fold_id}.pkl", "wb"))
    pickle.dump(extract_subset(neg_all, neg_train_ids), open(f"{out_dir}/negativedata_k{fold_id}.pkl", "wb"))
    pickle.dump(extract_subset(neg_all, neg_test_ids), open(f"{out_dir}/negativedata_test_k{fold_id}.pkl", "wb"))

    print(f"✅ Fold {fold_id} .pkl files saved to: {out_dir}")

if __name__ == "__main__":
    for k in range(1, 6):
        build_fold_pkl(
            fold_id=k,
            pos_pkl_path="esm2/positivedata549_esm2.pkl",
            neg_pkl_path="esm2/negativedata1670_esm2.pkl",
            split_dir="data/train_test_seqname",
            out_dir=f"esm2/fold{k}_pkl"
        )
