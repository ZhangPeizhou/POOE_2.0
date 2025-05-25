import os
import subprocess
import torch
import pickle
from tqdm import tqdm
from Bio import SeqIO
from collections import defaultdict

def split_sequence(seq, chunk_size=1000):
    return [seq[i:i+chunk_size] for i in range(0, len(seq), chunk_size)]

def extract_esm2_features(input_fastas, output_pkl):
    tmp_folder = "esm2/tmp_additional"
    os.makedirs(tmp_folder, exist_ok=True)

    chunks = defaultdict(list)
    full_lengths = {}

    # Step 1: Read sequences from all fasta files and split
    for fasta_path in input_fastas:
        for record in SeqIO.parse(fasta_path, "fasta"):
            name = record.id
            seq = str(record.seq)
            split_seqs = split_sequence(seq)
            full_lengths[name] = len(seq)
            for idx, chunk in enumerate(split_seqs):
                chunk_name = f"{name}_chunk{idx}"
                chunks[name].append(chunk_name)
                with open(os.path.join(tmp_folder, chunk_name + ".fasta"), "w") as f:
                    f.write(f">{chunk_name}\n{chunk}\n")

    # Step 2: Extract embeddings for all chunks using ESM2
    for chunk_list in tqdm(chunks.values(), desc="Extracting ESM2 embeddings"):
        for chunk_name in chunk_list:
            fasta_file = os.path.join(tmp_folder, chunk_name + ".fasta")
            output_file = os.path.join(tmp_folder, chunk_name + ".pt")

            subprocess.run([
                "python", "esm/extract.py",
                "facebook/esm2_t33_650M_UR50D",
                fasta_file,
                output_file,
                "--repr_layers", "33",
                "--include", "mean"
            ], check=True)

    # Step 3: Load and aggregate chunk embeddings
    feature_dict = {}
    for name, chunk_list in chunks.items():
        embeddings = []
        weights = []
        for chunk_name in chunk_list:
            pt_file = os.path.join(tmp_folder, chunk_name + ".pt")
            if not os.path.exists(pt_file):
                continue
            data = torch.load(pt_file)
            rep = data["representations"][33].squeeze(0)
            embeddings.append(rep)
            weights.append(rep.shape[0])
        if embeddings:
            stacked = torch.stack(embeddings)
            weights_tensor = torch.tensor(weights).float()
            weights_tensor /= weights_tensor.sum()
            weighted_avg = torch.sum(stacked * weights_tensor[:, None], dim=0)
            feature_dict[name] = weighted_avg.detach().numpy()

    # Step 4: Save final dictionary
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
