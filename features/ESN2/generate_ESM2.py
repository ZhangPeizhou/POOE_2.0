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

