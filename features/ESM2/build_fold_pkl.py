import os
import pickle

def load_seqnames(txt_file):
    with open(txt_file) as f:
        return [line.strip().lstrip(">") for line in f if line.strip()]

def extract_subset(pkl_data, id_list, role, fold_k, part):
    found = {}
    not_found = []

    for pid in id_list:
        if pid in pkl_data:
            found[pid] = pkl_data[pid]
        else:
            not_found.append(pid)

    if not_found:
        print(f"⚠️ Fold {fold_k} {role} {part}: {len(not_found)} not found in .pkl (sample: {not_found[:3]})")

    return found

def build_one_fold(k, pos_dict, neg_dict, txt_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    train_ids = load_seqnames(os.path.join(txt_dir, f"x_train1670_seqname_k{k}.txt"))
    test_ids  = load_seqnames(os.path.join(txt_dir, f"x_test1670_seqname_k{k}.txt"))

    pos_train = [id for id in train_ids if id in pos_dict]
    neg_train = [id for id in train_ids if id in neg_dict]

    pos_test = [id for id in test_ids if id in pos_dict]
    neg_test = [id for id in test_ids if id in neg_dict]

    print(f"\n[Fold {k}] Stats:")
    print(f"  ✅ Pos train: {len(pos_train)}, Pos test: {len(pos_test)}")
    print(f"  ✅ Neg train: {len(neg_train)}, Neg test: {len(neg_test)}")

    # 构造实际 embedding 子集
    pickle.dump(extract_subset(pos_dict, pos_train, "pos", k, "train"), open(f"{output_dir}/positivedata_k{k}.pkl", "wb"))
    pickle.dump(extract_subset(pos_dict, pos_test,  "pos", k, "test"),  open(f"{output_dir}/positivedata_test_k{k}.pkl", "wb"))
    pickle.dump(extract_subset(neg_dict, neg_train, "neg", k, "train"), open(f"{output_dir}/negativedata_k{k}.pkl", "wb"))
    pickle.dump(extract_subset(neg_dict, neg_test,  "neg", k, "test"),  open(f"{output_dir}/negativedata_test_k{k}.pkl", "wb"))

    print(f"✅ Fold {k} saved to: {output_dir}")

if __name__ == "__main__":
    # 读取已有 embedding
    with open("esm2/positivedata549_esm2.pkl", "rb") as f:
        pos_pkl = pickle.load(f)

    with open("esm2/negativedata1670_esm2.pkl", "rb") as f:
        neg_pkl = pickle.load(f)

    for k in range(1, 6):
        build_one_fold(
            k=k,
            pos_dict=pos_pkl,
            neg_dict=neg_pkl,
            txt_dir="data/train_test_seqname",
            output_dir=f"esm2/fold{k}_pkl"
        )
