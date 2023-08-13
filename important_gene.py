import numpy as np
import pandas as pd
from itertools import chain


def load_coexpressed_result(filename):
    coexpressed_result = np.zeros((genes, genes))
    s = open(filename)
    j = 0
    for line in s:
        seperate = line.split()
        coexpressed_result[j, :] = seperate
        j = j + 1
    s.close()
    coexpressed_result = coexpressed_result.astype(dtype=np.float64)
    return coexpressed_result


def load_gene_list(filename):
    gene_list = pd.read_csv(filename, header='infer', index_col=0)
    gene_list = np.array(gene_list)
    gene_list = list(chain.from_iterable(gene_list))
    print("The number of gene: ", len(gene_list))
    return gene_list


def calculate_predEdgeDF(coexpressed_result):
    matrix_calculate = np.zeros((genes, genes))
    for i in range(genes):
        row = coexpressed_result[i]
        mu, std = np.mean(row), np.std(row)
        if (mu > 0.7 and mu < 0.95 and std >= 0.005):
            d_value = (mu - row) / std
            d_value[d_value < 0] = 0
            matrix_calculate[i] = d_value

    predEdgesDict = {}
    cnt = 1
    for i in range(genes):
        for j in range(genes):
            if i == j or matrix_calculate[i][j] == 0:
                continue
            geneA_name = gene_list[i]
            geneB_name = gene_list[j]
            predEdgesDict[cnt] = [geneB_name, geneA_name, matrix_calculate[i][j]]
            cnt += 1
    predEdgesDF = pd.DataFrame.from_dict(predEdgesDict, orient='index', columns=['Gene1', 'Gene2', 'EdgeWeight'])
    print(predEdgesDF.shape)

    predEdgesDF_sorted = predEdgesDF.sort_values(by="EdgeWeight", ascending=False)
    length = 100
    edges_weight = predEdgesDF_sorted.iloc[0: length]
    return edges_weight

def sorted_gene(edges_weight):
    gene_cnt = {}
    for index, pred_edge in edges_weight.iterrows():
        gene1_name = pred_edge['Gene1']
        gene2_name = pred_edge['Gene2']
        if gene1_name not in gene_cnt.keys():
            gene_cnt[gene1_name] = 0
        gene_cnt[gene1_name] += pred_edge['EdgeWeight']
        if gene2_name not in gene_cnt.keys():
            gene_cnt[gene2_name] = 0
        gene_cnt[gene2_name] += pred_edge['EdgeWeight']
    genes_sort = sorted(gene_cnt.items(), key=lambda x: x[1], reverse=True)
    return genes_sort


if __name__ == '__main__':

    gene_file = "gene_list.csv"
    gene_list = load_gene_list(gene_file)
    genes = len(gene_list)
    coexpressed_file_list = ["coexpressed_result1.txt", "coexpressed_result2.txt", "coexpressed_result3.txt"]
    for coexpressed_file in coexpressed_file_list:
        coexpressed_result = load_coexpressed_result(coexpressed_file)
        pre_edges = calculate_predEdgeDF(coexpressed_result)
        genes_sort = sorted_gene(pre_edges)
        print(genes_sort[0: 50])
        print("------------------------------------------")