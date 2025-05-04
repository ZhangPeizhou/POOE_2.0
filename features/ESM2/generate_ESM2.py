import argparse
import os
import re
import torch
import pickle
import numpy as np
from Bio import SeqIO
import subprocess
import sys


def clean_sequence(seq):
    """
    Remove non-standard amino acids (only keep 20 standard: ACDEFGHIKLMNPQRSTVWY).
    """
    return re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', seq)


def prepare_fasta_for_esm2(input_fasta_path, output_fasta_path, max_seq_len=4096):
    """
    Read input fasta, clean illegal characters, split if needed, and save for ESM.
    """
    sequences = {rec.id: str(rec.seq) for rec in SeqIO.parse(input_fasta_path, "fasta")}
    with open(output_fasta_path, "w") as out_f:
        for seq_id, seq in sequences.items():
            seq_clean = clean_sequence(seq)
            if len(seq_clean) == 0:
                print(f"⚠️ Skipped {seq_id}: sequence cleaned to empty.")
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
    Call ESM2 extract.py to generate .pt files for embeddings.
    """
    os.makedirs(output_dir, exist_ok=True)
    script_path = "features/ESM2/extract.py"
    cmd = [
        sys.executable, script_path,
        model_name,
        input_fasta_path,
        output_dir,
        "--include", "mean",
        "--repr_layers", str(repr_layer)
    ]
    print(f"🚀 Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def extract_esm2_features(temp_dir, output_path):
    """
    Aggregate all .pt files from ESM output into a single .pkl file.
    Supports recombining chunked sequences using weighted average.
    """
    out_esm_dir = os.path.join(temp_dir, "out_esm")
    esm_mean = {}
    ids_embs = {}

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    for fname in os.listdir(out_esm_dir):
        if not fname.endswith(".pt"):
            continue
        if "__" in fname:
            k = fname.split("__")[0]
            ids_embs.setdefault(k, []).append(fname)
        else:
            ids_embs[fname.split(".pt")[0]] = [fname]

    for k, v in ids_embs.items():
        v = sorted(v)
        if len(v) == 1:
            tensor = torch.load(os.path.join(out_esm_dir, v[0]))["mean_representations"][33].numpy()
            esm_mean[k] = tensor
        else:
            tensors = []
            weights = []
            for fname in v:
                try:
                    length = int(fname.split("_")[1].split(".")[0])
                    tensor = torch.load(os.path.join(out_esm_dir, fname))["mean_representations"][33].numpy()
                    tensors.append(tensor)
                    weights.append(length)
                except Exception as e:
                    print(f"⚠️ Skipping {fname}: {e}")
                    continue
            if tensors:
                weights = np.array(weights, dtype=np.float32)
                weights /= weights.sum()
                weighted_avg = np.sum([t * w for t, w in zip(tensors, weights)], axis=0)
                esm_mean[k] = weighted_avg
  
    with open(output_path, "wb") as f:
        pickle.dump(esm_mean, f)

    print(f"✅ Saved: {output_path} with {len(esm_mean)} sequences")


def run_single_fasta(input_fasta, output_pkl, temp_dir):
    os.makedirs(temp_dir, exist_ok=True)
    input_fasta_for_esm = os.path.join(temp_dir, "input_for_esm.fasta")
    out_esm_dir = os.path.join(temp_dir, "out_esm")

    if os.path.exists(input_fasta_for_esm):
        os.remove(input_fasta_for_esm)
    if os.path.exists(out_esm_dir):
        for f in os.listdir(out_esm_dir):
            os.remove(os.path.join(out_esm_dir, f))

    prepare_fasta_for_esm2(input_fasta, input_fasta_for_esm)
    run_esm2_extraction(input_fasta_for_esm, out_esm_dir)
    extract_esm2_features(temp_dir, output_pkl)


def main():
    parser = argparse.ArgumentParser(description="Run ESM2 on one fasta file and save as .pkl")
    parser.add_argument("input_fasta", type=str, help="Input FASTA file path")
    parser.add_argument("output_pkl", type=str, help="Output .pkl file path")
    parser.add_argument("tmp_dir", type=str, help="Temporary directory")
    args = parser.parse_args()

    print(f"\n📂 Input:  {args.input_fasta}")
    print(f"📦 Output: {args.output_pkl}")
    print(f"🧪 Temp:   {args.tmp_dir}")
    run_single_fasta(args.input_fasta, args.output_pkl, args.tmp_dir)


if __name__ == "__main__":
    main()

'''
!python3.9 features/ESM2/generate_ESM2.py \
  data/training_data/positivedata549.fasta \
  features/ESM2/output_by_fasta/positivedata549.pkl \
  features/ESM2/tmp_cache/
'''