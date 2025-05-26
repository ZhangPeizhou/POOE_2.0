import os
import torch
import pickle
from tqdm import tqdm
from Bio import SeqIO
from collections import defaultdict
import esm

def split_sequence(seq, chunk_size=1000):
    return [seq[i:i+chunk_size] for i in range(0, len(seq), chunk_size)]

def extract_esm2_features(input_fastas, output_pkl):
    tmp_folder = "esm2/tmp_additional"
    os.makedirs(tmp_folder, exist_ok=True)

    # 加载模型
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model.eval()
    batch_converter = alphabet.get_batch_converter()

    chunks = defaultdict(list)
    valid_aas = set("ACDEFGHIKLMNPQRSTVWY")  # 合法氨基酸

    # Step 1: 读取序列，清洗，并分块
    for fasta_path in input_fastas:
        for record in SeqIO.parse(fasta_path, "fasta"):
            name = record.id
            raw_seq = str(record.seq)
            seq = ''.join([aa for aa in raw_seq if aa in valid_aas])
            if len(seq) == 0:
                print(f"⚠️ Skipped empty or invalid sequence: {name}")
                continue
            split_seqs = split_sequence(seq)
            for idx, chunk in enumerate(split_seqs):
                chunk_name = f"{name}_chunk{idx}"
                chunks[name].append((chunk_name, chunk))

    # Step 2: 提取 embedding
    feature_dict = {}
    for name, chunk_list in tqdm(chunks.items(), desc="Extracting ESM2 embeddings"):
        embeddings = []
        weights = []
        batch_data = [(chunk_name, seq) for chunk_name, seq in chunk_list]
        batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)
        token_representations = results["representations"][33]

        for i, (_, seq) in enumerate(batch_data):
            rep = token_representations[i, 1:len(seq)+1].mean(0)
            embeddings.append(rep)
            weights.append(len(seq))

        weights_tensor = torch.tensor(weights).float()
        weights_tensor /= weights_tensor.sum()
        stacked = torch.stack(embeddings)
        weighted_avg = torch.sum(stacked * weights_tensor[:, None], dim=0)
        feature_dict[name] = weighted_avg.numpy()

    # Step 3: 保存输出
    with open(output_pkl, "wb") as f:
        pickle.dump(feature_dict, f)

    print(f"✅ Saved {len(feature_dict)} embeddings to {output_pkl}")

if __name__ == "__main__":
    input_fastas = [
        "data/additional_data/Additional_test_29.fasta",
        "data/additional_data/Additional_test_38.fasta" 
    ]
    output_pkl = "esm2/additional_test_esm2.pkl"
    extract_esm2_features(input_fastas, output_pkl)
