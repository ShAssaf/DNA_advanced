import os
import re
from Bio import SeqIO


def load_data(directory):
    data = {}
    for filename in os.listdir(directory):
        if filename.endswith(".fasta"):
            lineage = filename.replace(".fasta", "")
            sequences = list(SeqIO.parse(os.path.join(directory, filename), "fasta"))
            data[lineage] = [str(seq.seq) for seq in sequences]
    return data


def filter_nonstandard(sequence):
    filtered_sequence = re.sub(r'[^ACGT]', '', sequence)
    return filtered_sequence