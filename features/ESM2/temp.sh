#!/bin/bash

# Example usage:
# bash ./temp.sh esm2_t33_650M_UR50D ../../data/training_data/positivedata549.fasta ./tmp_esm2 ./positivedata549_esm2.pkl

MODEL=$1
INPUT_FASTA=$2
TMP_DIR=$3
SAVE_PATH=$4

mkdir -p $TMP_DIR

python generate_ESM2.py $MODEL $INPUT_FASTA $TMP_DIR \
    --include mean \
    --repr_layers 33 \
    --truncation_seq_length 4000 \
    --save_file $SAVE_PATH
