import numpy as np
import pandas as pd
from itertools import permutations, combinations
from sklearn.metrics import precision_recall_curve, roc_curve, auc


def generate_trueEdges(filename):
    trueEdgesDict = {}
    s = open(filename)
    cnt = 0
    for line in s:
        separation = line.split(',')
        geneA_name, geneB_name, label = separation[0], separation[1], separation[2]
        label = label.strip()
        if label == '+' or label == '-':
            trueEdgesDict[cnt] = [geneA_name, geneB_name]
            cnt += 1
    s.close()
    trueEdgesDF = pd.DataFrame.from_dict(trueEdgesDict, orient='index', columns=['Gene1', 'Gene2'])
    return trueEdgesDF


def load_gene_list(filename):
    gene_list = []
    s = open(filename)
    for line in s:
        gene = line.strip()
        gene_list.append(gene)
    return gene_list


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


def calculate_predEdgeDF():
    matrix_calculate = np.zeros((genes, genes))
    for i in range(genes):
        row = coexpressed_result[i]
        mu, std = np.mean(row), np.std(row)
        if std == 0:
            continue
        d_value = (mu - row) / std
        d_value[d_value < 0] = 0
        matrix_calculate[i] = d_value
    predEdgesDict = {}
    cnt = 1
    for i in range(genes):
        for j in range(genes):
            geneA_name = gene_list[i]
            geneB_name = gene_list[j]
            predEdgesDict[cnt] = [geneB_name, geneA_name, matrix_calculate[i][j]]
            cnt += 1
    predEdgesDF = pd.DataFrame.from_dict(predEdgesDict, orient='index', columns=['Gene1', 'Gene2', 'EdgeWeight'])
    return predEdgesDF


def computeAUC():
    # if directed
    possibleEdges = list(permutations(np.unique(trueEdgesDF.loc[:, ['Gene1', 'Gene2']]),
                                      r=2))
    TrueEdgeDict = {'|'.join(p): 0 for p in possibleEdges}
    PredEdgeDict = {'|'.join(p): 0 for p in possibleEdges}

    # Compute TrueEdgeDict Dictionary
    # 1 if edge is present in the ground-truth
    # 0 if edge is not present in the ground-truth
    for key in TrueEdgeDict.keys():
        if len(trueEdgesDF.loc[(trueEdgesDF['Gene1'] == key.split('|')[0]) &
                               (trueEdgesDF['Gene2'] == key.split('|')[1])]) > 0:
            TrueEdgeDict[key] = 1

    # Compute PredEdgeDict Dictionary
    for key in PredEdgeDict.keys():
        subDF = predEdgeDF.loc[(predEdgeDF['Gene1'] == key.split('|')[0]) &
                               (predEdgeDF['Gene2'] == key.split('|')[1])]
        if len(subDF) > 0:
            PredEdgeDict[key] = np.abs(subDF.EdgeWeight.values[0])

    # Combine into one dataframe
    # to pass it to sklearn
    outDF = pd.DataFrame([TrueEdgeDict, PredEdgeDict]).T
    outDF.columns = ['TrueEdges', 'PredEdges']

    fpr, tpr, thresholds = roc_curve(y_true=outDF['TrueEdges'],
                                     y_score=outDF['PredEdges'], pos_label=1)

    prec, recall, _ = precision_recall_curve(y_true=outDF['TrueEdges'],
                                             probas_pred=outDF['PredEdges'], pos_label=1)

    AUPRC = auc(recall, prec)
    AUROC = auc(fpr, tpr)
    return AUPRC, AUROC


if __name__ == '__main__':
    genes = 8  # BF: 7, TF:8, gsd:19
    gene_list_file = "dyn-TF/gene_list.txt"
    gene_list = load_gene_list(gene_list_file)

    AUPRC_list = []
    AUROC_list = []
    for i in range(1, 4):
        trueEdgesDF_file = "dyn-TF/refNetwork" + str(i) + ".csv"
        trueEdgesDF = generate_trueEdges(trueEdgesDF_file)
        coexpressed_ntf_file = "output/coexpressed_result" + str(i) + ".txt"
        coexpressed_result = load_coexpressed_result(coexpressed_ntf_file)
        predEdgeDF = calculate_predEdgeDF()
        AUPRC, AUROC = computeAUC()
        AUPRC_list.append(AUPRC)
        AUROC_list.append(AUROC)
    print("AUROC: ")
    for auroc in AUROC_list:
        print(" ", format(auroc, '.3f'), end="")
    print(" ")
    print("AUPRC: ")
    for auprc in AUPRC_list:
        print(" ", format(auprc, '.3f'), end="")
