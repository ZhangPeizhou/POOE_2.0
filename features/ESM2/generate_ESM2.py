import argparse
import os
from Bio import SeqIO
import torch
import pickle
import numpy as np
from tqdm import tqdm
import esm


def get_esm2_model():
    """
    Load the pretrained ESM-2 model and its tokenizer (alphabet).
    This version uses esm2_t33_650M_UR50D, which has 650M parameters.
    """
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    return model, batch_converter

def prepare_fasta_for_esm2(input_fasta_path, output_fasta_path, max_seq_len=4096):
    """
    Reads sequences from an input FASTA file and writes them to a new FASTA file.
    If any sequence is longer than `max_seq_len`, it will be split into chunks.
    """
    sequences = {rec.id: str(rec.seq) for rec in SeqIO.parse(input_fasta_path, "fasta")}

    with open(output_fasta_path, "w") as out_f:
        for seq_id, seq in sequences.items():
            if len(seq) > max_seq_len:
                # Split long sequences into sub-segments
                for i in range((len(seq) - 1) // max_seq_len + 1):
                    sub_seq = seq[i * max_seq_len : (i + 1) * max_seq_len]
                    sub_id = f"{seq_id}__{i}_{len(sub_seq)}"
                    out_f.write(f">{sub_id}\n{sub_seq}\n")
            else:
                # Write normally if not too long
                out_f.write(f">{seq_id}\n{seq}\n")

def run_esm2_extraction(input_fasta_path, output_dir, model_name="esm2_t33_650M_UR50D", repr_layer=33):
    """
    Runs the ESM-2 extraction command using the given model.
    Outputs embeddings to `output_dir` folder using mean pooling at the specified layer.
    """
    os.makedirs(output_dir, exist_ok=True)
    cmd = f"python features/ESM2/extract.py {model_name} {input_fasta_path} {output_dir} --include mean --repr_layers {repr_layer}"
    print(f"Running: {cmd}")
    os.system(cmd)

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
    parser = argparse.ArgumentParser(description='Generate ESM-2 embeddings of proteins')
    parser.add_argument("input", type=str, help='Input FASTA file')
    parser.add_argument("output", type=str, help='Output pickle file')
    parser.add_argument("tmp_dir", type=str, default="./cache/", help='Temporary directory')

    args = parser.parse_args()

    run_esm2_emb_model(args.input, args.tmp_dir)
    extract_esm2_features(args.tmp_dir, args.output)

if __name__ == "__main__":
    main()


'''
!python3.9 features/ESM2/generate_ESM2.py \
    data/training_data/positivedata549.fasta \
    features/ESM2/output_embeds.pkl \
    features/ESM2/tmp_cache/
'''
