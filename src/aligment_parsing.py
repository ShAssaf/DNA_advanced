from Bio import pairwise2
from Bio.Seq import Seq
from tqdm import tqdm
from src.common_func import load_data


def calculate_alignment_score(seq1, seq2):
    alignments = pairwise2.align.globalxx(seq1, seq2)
    best_alignment = alignments[0]
    alignment_score = best_alignment.score
    return alignment_score


def average_alignment_score(sequence_list):
    total_score = 0
    num_pairs = 0

    # Use tqdm for both loops
    for i in tqdm(range(5), desc="Sequence Pair"):
        for j in tqdm(range(i + 1, 5), desc="Calculating Alignment", leave=False):
            seq1 = Seq(sequence_list[i])
            seq2 = Seq(sequence_list[j])
            alignment_score = calculate_alignment_score(seq1, seq2)
            total_score += alignment_score
            num_pairs += 1

    if num_pairs == 0:
        return 0

    average_score = total_score / num_pairs
    return average_score


def calc_alignment_score_for_each_variant(data):
    alignment_scores = {}
    for lineage, sequences in data.items():
        alignment_scores[lineage] = average_alignment_score(sequences)

    return alignment_scores

if __name__ == '__main__':
    data_directory = '../data/50at100'
    print("loading data...")
    data = load_data(data_directory)
    print("calculating alignment scores")
    print(calc_alignment_score_for_each_variant(data))
    print("done")
