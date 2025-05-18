import argparse
import os
from Bio import SeqIO
import torch
import pickle
import numpy as np
from tqdm import tqdm
import esm

def preprocess_sequences(input_path, tmp_dir, max_len=1000):
    os.makedirs(tmp_dir, exist_ok=True)
    fasta_path = os.path.join(tmp_dir, "input_for_esm2.fasta")
    seqs = {rec.id: str(rec.seq) for rec in SeqIO.parse(input_path, "fasta")}

    with open(fasta_path, "w") as f:
        for k, v in seqs.items():
            if len(v) > max_len:
                for i in range(len(v) // max_len + 1):
                    sub_v = v[i*max_len : (i+1)*max_len]
                    if len(sub_v) == 0:
                        continue
                    sub_k = f"{k}__{i}_{len(sub_v)}"
                    f.write(f">{sub_k}\n{sub_v}\n")
            else:
                f.write(f">{k}\n{v}\n")
    return fasta_path

def extract_embedding(fasta_file, tmp_dir):
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    sequences = list(SeqIO.parse(fasta_file, "fasta"))
    output_dir = os.path.join(tmp_dir, "out_esm2")
    os.makedirs(output_dir, exist_ok=True)

    for i in tqdm(range(len(sequences))):
        name = sequences[i].id
        seq = str(sequences[i].seq)
        batch_labels, batch_strs, batch_tokens = batch_converter([(name, seq)])
        if torch.cuda.is_available():
            batch_tokens = batch_tokens.cuda()
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)
        token_reps = results["representations"][33]
        emb = token_reps[0, 1:len(seq)+1].mean(0).cpu().numpy()

        torch.save({'mean_representations': {33: torch.tensor(emb)}}, os.path.join(output_dir, f"{name}.pt"))

def merge_embedding(tmp_dir, output_file):
    esm_dir = os.path.join(tmp_dir, "out_esm2")
    ids = {}
    for f_name in os.listdir(esm_dir):
        if "__" in f_name:
            k = f_name.split("__")[0]
            ids.setdefault(k, []).append(f_name)
        else:
            ids[f_name.split(".pt")[0]] = [f_name]
    for k in ids:
        ids[k] = sorted(ids[k])

    merged = {}
    for k, files in tqdm(ids.items()):
        if len(files) == 1:
            vec = torch.load(os.path.join(esm_dir, files[0]))['mean_representations'][33].numpy()
            merged[k] = vec
        else:
            vecs, lens = [], []
            for f in files:
                vec = torch.load(os.path.join(esm_dir, f))['mean_representations'][33].numpy()
                length = int(f.split("__")[1].split("_")[1].split(".")[0])
                vecs.append(vec)
                lens.append(length)
            weights = np.array(lens, dtype=np.float32)
            weights = weights / weights.sum()
            merged[k] = np.sum([w * v for w, v in zip(weights, vecs)], axis=0)

    with open(output_file, "wb") as f:
        pickle.dump(merged, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name")  # ignored
    parser.add_argument("input_fasta")
    parser.add_argument("tmp_dir")
    parser.add_argument("--include", default="mean")
    parser.add_argument("--repr_layers", default=33, type=int)
    parser.add_argument("--truncation_seq_length", default=4000, type=int)
    parser.add_argument("--save_file", required=True)
    args = parser.parse_args()

    fasta_path = preprocess_sequences(args.input_fasta, args.tmp_dir, max_len=args.truncation_seq_length)
    extract_embedding(fasta_path, args.tmp_dir)
    merge_embedding(args.tmp_dir, args.save_file)

if __name__ == "__main__":
    main()
