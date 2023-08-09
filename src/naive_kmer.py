import numpy as np
import pandas as pd
from tqdm import tqdm

from src.common_func import load_data




def Kmers_funct(seq, size):
    return [seq[x:x + size].upper() for x in range(len(seq) - size + 1)]


def data_into_kmers_count(data):
    k_mers_count_dict = {}
    all_combs = generate_all_combs(K)
    k_mer_def = {i: 0 for i in all_combs.split()}
    for lineage, sequences in tqdm(data.items()):
        k_mers_vectors = []
        for sequence in sequences:
            k_mer_dict = k_mer_def.copy()
            for kmer in Kmers_funct(sequence, size=K):
                k_mer_dict[kmer] += 1
            k_mers_vectors.append([k_mer_dict[i] for i in sorted(k_mer_dict.keys())])

        k_mers_count_dict[lineage] = np.array(k_mers_vectors)
    return k_mers_count_dict


def generate_all_combs(K=4):
    from itertools import product
    li = ['A', 'T', 'G', 'C']
    combinations = []
    for comb in product(li, repeat=K):
        combinations.append(''.join(comb))
    return ' '.join(combinations)


data_directory = './data/common'
data = load_data(data_directory)
mer_dict_acc = {}


for K in range(4, 11):

    # data = {k:[i[0]] for k,i in data.items()}
    k_mers_count_dict = data_into_kmers_count(data)

    # save kmer_count_dict
    # for lineage, kmer_vector_array in k_mers_count_dict.items():
    #     np.save(f'./data/kmers_dict/{lineage}.npy', kmer_vector_array)
    #
    # # load kmer_count_dict
    # k_mers_count_dict = {}
    # for lineage in data.keys():
    #     k_mers_count_dict[lineage] = np.load(f'./data/kmers_dict/{lineage}.npy')

    # for each lineage, save the last 25 sequences as test data
    lineage_average_kmer_dict = {}
    test_dict = {}
    for lineage, kmer_vector_array in k_mers_count_dict.items():
        kmer_vector_array = kmer_vector_array[1:-25, :]
        test_dict[lineage] = kmer_vector_array[-25:, :]
        lineage_average_kmer_dict[lineage] = np.mean(kmer_vector_array, axis=0)

    k_accuracy = []
    for lineage, kmer_vector_array in test_dict.items():

        for i in range(25):
            res = []
            for lin in lineage_average_kmer_dict.keys():
                res.append((lin, np.linalg.norm(kmer_vector_array[i, :] - lineage_average_kmer_dict[lin])))
            # print(lineage, np.argmin(res))
            res = sorted(res, key=lambda x: x[1])
            for j in range(len(res)):
                if res[j][0] == lineage:
                    k_accuracy.append(j)
                    break
    print(K,"-Mer accuracy")
    top_k_dict = {}
    for i in range(1,6):
        print(f"top {i} accuracy ", "{:.2f}".format(len([x for x in k_accuracy if x < i]) / len(k_accuracy)))
        top_k_dict[i] = len([x for x in k_accuracy if x < i]) / len(k_accuracy)
    mer_dict_acc[K] = top_k_dict

    pd.DataFrame(mer_dict_acc).round(2).to_csv(f'./data/mer_dict_acc_{K}.csv')