import os
import psutil
import time
import torch
import math
import numpy as np
import pandas as pd
import scanpy as sc
import scanpy.external as sce
import anndata
from pathlib import Path
from sklearn.metrics import pairwise_distances, calinski_harabasz_score
from scipy.sparse import issparse, isspmatrix_csr, csr_matrix, spmatrix
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.spatial import distance
import scanpy as sc

from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from typing import Union, Callable

from sklearn import metrics
from sklearn.metrics import silhouette_score
def read_10X_Visium(path, 
    genome=None,
    count_file='filtered_feature_bc_matrix.h5', 
    library_id=None, 
    load_images=True, 
    quality='hires',
    image_path = None):
    adata = sc.read_visium(path, 
                        genome=genome,
                        count_file=count_file,
                        library_id=library_id,
                        load_images=load_images,)
    adata.var_names_make_unique()

    if library_id is None:
        library_id = list(adata.uns["spatial"].keys())[0]
    if quality == "fulres":
        image_coor = adata.obsm["spatial"]
        img = plt.imread(image_path, 0)
        adata.uns["spatial"][library_id]["images"]["fulres"] = img
    else:
        scale = adata.uns["spatial"][library_id]["scalefactors"][
            "tissue_" + quality + "_scalef"]
        image_coor = adata.obsm["spatial"] * scale
    adata.obs["imagecol"] = image_coor[:, 0]
    adata.obs["imagerow"] = image_coor[:, 1]
    adata.uns["spatial"][library_id]["use_quality"] = quality
    return adata

def refine(sample_id, pred, dis, shape="hexagon"):
    refined_pred=[]
    pred=pd.DataFrame({"pred": pred}, index=sample_id)
    dis_df=pd.DataFrame(dis, index=sample_id, columns=sample_id)
    if shape=="hexagon":
        num_nbs=6 
    elif shape=="square":
        num_nbs=4
    else:
        print("Shape not recongized, shape='hexagon' for Visium data, 'square' for ST data.")
    for i in range(len(sample_id)):
        index=sample_id[i]
        dis_tmp=dis_df.loc[index, :].sort_values()
        nbs=dis_tmp[0:num_nbs+1]
        nbs_pred=pred.loc[nbs.index, "pred"]
        self_pred=pred.loc[index, "pred"]
        v_c=nbs_pred.value_counts()
        if (v_c.loc[self_pred]<num_nbs/2) and (np.max(v_c)>num_nbs/2):
            refined_pred.append(v_c.idxmax())
        else:           
            refined_pred.append(self_pred)

    return refined_pred
def create_adjacency_matrix(cluster_labels):
    n = len(cluster_labels)
    K = len(np.unique(cluster_labels))
    adj_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            if cluster_labels[i] == cluster_labels[j]:
                adj_matrix[i][j] = 1
                adj_matrix[j][i] = 1
            else:
                adj_matrix[i][j] = 0
                adj_matrix[j][i] = 0               

    return adj_matrix
def priori_cluster(
		adata,
		n_domains,
		):
		for res in sorted(list(np.arange(0.1,1, 0.01)), reverse=True):
			sc.tl.leiden(adata, random_state=0, resolution=res)
			count_unique_leiden = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
			if count_unique_leiden == n_domains:
				break
		print("Best resolution: ", res)
		return res
def cluster(adata,X,df_meta,n_domains,refined=False):
        adata.obsm["pca_data"] = X
        sc.pp.neighbors(adata, n_neighbors=6,use_rep="pca_data")
        res = priori_cluster(adata, n_domains)
        sc.tl.leiden(adata, key_added="domain", resolution=res)
        if refined:
            for i in range(3):
                adj_2d = distance.cdist(adata.obsm['spatial'], adata.obsm['spatial'], 'euclidean')
                refined_pred = refine(sample_id=adata.obs.index.tolist(), pred=adata.obs["domain"].tolist(),
                                    dis=adj_2d, shape="hexagon")
                adata.obs["domain"] = refined_pred

        df_meta['clusters'] = adata.obs['domain'].tolist()
        df_meta = df_meta[~pd.isnull(df_meta['layer_guess_reordered'])]
        ARI = metrics.adjusted_rand_score(df_meta['layer_guess_reordered'], df_meta['clusters'])
        NMI = metrics.normalized_mutual_info_score(df_meta['layer_guess_reordered'],df_meta['clusters'])
        print("ARI:{:.4f}".format(ARI))
        print("NMI:{:.4f}".format(NMI))


        return adata.obs["domain"].tolist(),ARI
def plot_domains(adata, 
            data_name,
            n=0,
            img_key=None, 
            color="domain",
            show=False,
            legend_loc='right margin',
            legend_fontsize='x-large',
            size=1.6,
            dpi=300):
    if isinstance(data_name, str):
        sc.pl.spatial(adata, img_key=img_key, color=color, show=show, 
                    legend_loc=legend_loc, legend_fontsize=legend_fontsize, size=size)
        save_path_figure = Path(os.path.join("/home/zhangzhihao/SpaGCN/SpaGCN-master/tutorial/data/151673/V1_Mouse_Brain_Sagittal_Posterior_spatial", "Figure1", data_name))
        save_path_figure.mkdir(parents=True, exist_ok=True)
        plt.savefig(os.path.join(save_path_figure,f'{data_name}_domains_V1_'+str(n)+'.pdf'), bbox_inches='tight', dpi=dpi)
    else:
        pass 
def found_gene_count(adata,cell,gene):
        # 指定特定类型的细胞
    cell_type = cell  # 替换为你要筛选的细胞类型

    # 指定要计算平均表达值的基因
    gene_name = gene  # 替换为你要计算的基因名称

    # 使用布尔索引选取特定类型的细胞
    selected_cells = adata.obs['labels'] == cell_type  # 替换为你的细胞类型列名称

    # 提取选定细胞的基因表达数据
    selected_gene_expression = adata[selected_cells].X

    # 找到指定基因的索引
    gene_index = adata.var_names.get_loc(gene_name)

    # 计算平均表达值
    average_expression = np.mean(selected_gene_expression[:, gene_index])

    print(f"平均表达值: {average_expression}")
def found_bigest_gene_expression(adata):
    from collections import defaultdict
    # 提取细胞类型信息
    cell_types = adata.obs['labels']  # 替换为你的细胞类型列名称

    # 初始化字典以存储每种细胞类型的平均表达值
    cell_type_avg_expression = defaultdict(list)

    # 遍历每个细胞
    for idx, cell_type in enumerate(cell_types):
        gene_expression = adata.X[idx, :]  # 提取当前细胞的基因表达数据
        cell_type_avg_expression[cell_type].append(gene_expression)  # 存储到对应的细胞类型中

    # 计算每种细胞类型中基因的平均表达值
    for cell_type, gene_expressions in cell_type_avg_expression.items():
        gene_expressions = np.vstack(gene_expressions)
        avg_expression = np.mean(gene_expressions, axis=0)
        cell_type_avg_expression[cell_type] = avg_expression

    # 找到每种细胞类型平均值最大的六个基因的索引
    top_genes = {}
    for cell_type, avg_expression in cell_type_avg_expression.items():
        top_genes_indices = np.argsort(avg_expression)[-6:]
        top_gene_names = adata.var_names[top_genes_indices]
        top_gene_avg_expression = avg_expression[top_genes_indices]
        top_genes[cell_type] = list(zip(top_gene_names, top_gene_avg_expression))

    # 输出每种细胞类型平均值最大的六个基因的基因名和平均表达值
    for cell_type, genes in top_genes.items():
        print(f"细胞类型: {cell_type}")
        for gene, avg_expression in genes:
            print(f"基因名: {gene}, 平均表达值: {avg_expression:.2f}")
        print()
def symmetric_normalize_adjacency(adjacency_matrix):
    row_sum = torch.sparse.sum(adjacency_matrix, dim=1).to_dense()
    degree_matrix_sqrt_inv = torch.diag(torch.pow(row_sum, -0.5))
    normalized_adjacency = degree_matrix_sqrt_inv @ adjacency_matrix @ degree_matrix_sqrt_inv
    return normalized_adjacency
def sim2adj(sim,num):
    top_indices = np.argsort(sim, axis=1)[:,-num:]

    # 创建一个与pcc_matrix相同形状的全零矩阵
    binary_matrix = np.zeros_like(sim)

    # 将最大的4个值对应的位置置为1
    for i, indices in enumerate(top_indices):
        binary_matrix[i, indices] = 1
    return binary_matrix