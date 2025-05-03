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
    cmd = f"python extract.py {model_name} {input_fasta_path} {output_dir} --include mean --repr_layers {repr_layer}"
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

