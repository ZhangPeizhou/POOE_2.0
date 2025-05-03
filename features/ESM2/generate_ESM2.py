import argparse
import os
from Bio import SeqIO
import torch
import pickle
import numpy as np
from tqdm import tqdm
import esm
import subprocess
import sys
import re


def get_esm2_model():
    """
    Load the pretrained ESM-2 model and its tokenizer (alphabet).
    This version uses esm2_t33_650M_UR50D, which has 650M parameters.
    """
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    return model, batch_converter


def clean_sequence(seq):
    """
    Remove any non-standard amino acid characters (e.g., '*', 'X', etc.)
    Only keep 20 standard amino acids: A, C, D, E, F, G, H, I, K, L, M,
    N, P, Q, R, S, T, V, W, Y
    """
    return re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', seq)


def prepare_fasta_for_esm2(input_fasta_path, output_fasta_path, max_seq_len=4096):
    """
    Reads sequences from an input FASTA file and writes them to a new FASTA file.
    If any sequence is longer than `max_seq_len`, it will be split into chunks.
    """
    sequences = {rec.id: str(rec.seq) for rec in SeqIO.parse(input_fasta_path, "fasta")}

    with open(output_fasta_path, "w") as out_f:
        for seq_id, seq in sequences.items():
            seq_clean = clean_sequence(seq)  # 清洗非法字符
            if len(seq_clean) == 0:
                print(f"Warning: sequence {seq_id} was entirely invalid and removed.")
                continue

            if len(seq_clean) > max_seq_len:
                for i in range((len(seq_clean) - 1) // max_seq_len + 1):
                    sub_seq = seq_clean[i * max_seq_len : (i + 1) * max_seq_len]
                    sub_id = f"{seq_id}__{i}_{len(sub_seq)}"
                    out_f.write(f">{sub_id}\n{sub_seq}\n")
            else:
                out_f.write(f">{seq_id}\n{seq_clean}\n")


def run_esm2_extraction(input_fasta_path, output_dir, model_name="esm2_t33_650M_UR50D", repr_layer=33):
    """
    Runs the ESM-2 extraction command using the given model.
    Outputs embeddings to `output_dir` folder using mean pooling at the specified layer.
    """
    os.makedirs(output_dir, exist_ok=True)
    script_path = "features/ESM2/extract.py"
    cmd = [
        sys.executable, script_path,  # 使用当前解释器
        model_name,
        input_fasta_path,
        output_dir,
        "--include", "mean",
        "--repr_layers", str(repr_layer)
    ]
    print("Running command:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    

def run_esm2_emb_model(seq_file, temp_dir):
    """
    Master function that calls the above two sub-functions in order:
    1. Prepares the input fasta for ESM2 (with chunking).
    2. Runs the ESM2 embedding extraction script.
    """
    os.makedirs(temp_dir, exist_ok=True)
    input_fasta_for_esm = os.path.join(temp_dir, "input_for_esm.fasta")
    output_dir_for_esm = os.path.join(temp_dir, "out_esm")

    prepare_fasta_for_esm2(seq_file, input_fasta_for_esm, max_seq_len=4096)
    run_esm2_extraction(input_fasta_for_esm, output_dir_for_esm)

def extract_esm2_features(temp_dir, out_file):
    """
    Extracts mean-pooled representations from saved .pt files in temp_dir/out_esm
    and saves them as a pickle file.
    If a sequence was split into chunks, re-averages using weighted sum.
    """
    esm_mean = {}
    out_esm_dir = os.path.join(temp_dir, "out_esm")

    # Group all chunks by original sequence ID
    ids_embs = {}
    for f_name in os.listdir(out_esm_dir):
        if "__" in f_name:
            k = f_name.split("__")[0]
            ids_embs.setdefault(k, []).append(f_name)
        else:
            ids_embs[f_name.split(".pt")[0]] = [f_name]

    for k, v in ids_embs.items():
        v = sorted(v)
        if len(v) == 1:
            tensor = torch.load(os.path.join(out_esm_dir, v[0]))["mean_representations"][33].numpy()
            esm_mean[k] = tensor
        else:
            # Handle chunked sequences with length-weighted average
            tensors = []
            weights = []
            for fname in v:
                tensor = torch.load(os.path.join(out_esm_dir, fname))["mean_representations"][33].numpy()
                length = int(fname.split("_")[1].split(".")[0])
                tensors.append(tensor)
                weights.append(length)
            weights = np.array(weights, dtype=np.float32)
            weights /= weights.sum()
            weighted_avg = np.sum([t * w for t, w in zip(tensors, weights)], axis=0)
            esm_mean[k] = weighted_avg

    with open(out_file, "wb") as f:
        pickle.dump(esm_mean, f)

def main():
    parser = argparse.ArgumentParser(description='Batch generate ESM-2 embeddings for multiple FASTA files')
    parser.add_argument("input_dir", type=str, help='Input directory containing FASTA files')
    parser.add_argument("output_dir", type=str, help='Output directory for pickle files')
    parser.add_argument("tmp_dir", type=str, default="./cache/", help='Temporary directory for intermediate files')
    args = parser.parse_args()

    fasta_files = [f for f in os.listdir(args.input_dir) if f.endswith(".fasta")]
    os.makedirs(args.output_dir, exist_ok=True)

    for fasta_file in fasta_files:
        input_path = os.path.join(args.input_dir, fasta_file)
        output_name = fasta_file.replace(".fasta", ".pkl")
        output_path = os.path.join(args.output_dir, output_name)

        print(f"\nProcessing: {fasta_file}")
        run_esm2_emb_model(input_path, args.tmp_dir)
        extract_esm2_features(args.tmp_dir, output_path)



if __name__ == "__main__":
    main()

# Process training data
'''
!python3.9 features/ESM2/generate_ESM2.py \
    data/training_data \
    features/ESM2/output_pkls \
    features/ESM2/tmp_cache/
'''
# Process additional data
'''
!python3.9 features/ESM2/generate_ESM2.py \
    data/additional_data/Additional_test_29.fasta \
    features/ESM2/pkl_additional_29.pkl \
    features/ESM2/tmp_add_29/
'''