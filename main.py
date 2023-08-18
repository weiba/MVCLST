import os 
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from get_data import run
import scanpy as sc
from sklearn import metrics
from scipy.spatial import distance
from utils import *
from torch_geometric.nn import HypergraphConv
from MVCLST import train
import anndata
from scipy.stats import zscore
from utils_copy import clustering
from sklearn.metrics import silhouette_score
from augment import *

sample_list = ['151673']#'151509','151510','151669','151670','151671','151672','151673','151674','151675','151676'
for sample in sample_list:
	data_path = "/home/zhangzhihao/MVCLST/data/151673" #### to your path
	data_name = sample
	save_path = data_path+"/"+sample+"/" #### save path
	save_path_figure = Path(os.path.join(save_path, "Figure", data_name))
	save_path_figure.mkdir(parents=True, exist_ok=True)
	if data_name in ['151669','151670','151671','151672']:
		n_domains = 5
	else:
		n_domains = 7
	data = run(save_path = save_path, 
		platform = "Visium",
		pca_n_comps = 128,
		pre_epochs = 800,
		vit_type='vit_b',#'vit'
		)
	df_meta = pd.read_csv(data_path+'/'+data_name+'/metadata.tsv', sep='\t')
	adata =data._get_adata(data_path, data_name)
	adata = anndata.read_h5ad(save_path+"/"+sample+".h5ad")
	adata = data._get_augment(adata, adjacent_weight = 1, neighbour_k =6)
	adata1=adata.copy()
	adata2=adata.copy()
	adata.X = adata.obsm["augment_gene_data"].astype(float)
	sc.pp.filter_genes(adata, min_cells=3)
	sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
	adata_X = sc.pp.normalize_total(adata, target_sum=1, exclude_highly_expressed=True, inplace=False)['X']
	adata_X = sc.pp.log1p(adata_X)
	adata_X = sc.pp.scale(adata_X)
	inputs1 = sc.pp.pca(adata_X, n_comps=128)
	inputs = sc.pp.pca(adata_X, n_comps=1000)
	cluster_label,_=cluster(adata,inputs1,df_meta,n_domains)
	cluster_adj=create_adjacency_matrix(cluster_label)
	adj_augment=sim2adj(adata.obsm["weights_matrix_all"],6)
	adj_pure=sim2adj(adata.obsm["weights_matrix_nomd"],6)
	adj_pure=cluster_adj*adj_pure
	adata1.obsm['weights_matrix_all']=adj_pure
	adata1=find_adjacent_spot(
	adata1,
	use_data = "raw",
	neighbour_k = 4,
	weights='weights_matrix_all',
	verbose = False,
	)
	adata1=augment_gene_data(
	adata1,
	use_data = "raw",
	adjacent_weight = 1,
	)
	adata1.X = adata1.obsm["augment_gene_data"].astype(float)
	adata1_X = sc.pp.normalize_total(adata1, target_sum=1, exclude_highly_expressed=True, inplace=False)['X']
	adata1_X = sc.pp.log1p(adata1_X)
	adata1_X = sc.pp.scale(adata1_X)
	inputs2 = sc.pp.pca(adata1_X, n_comps=1000)
	X=inputs.copy()
	X2=inputs2.copy()
	X=torch.tensor(X,dtype=torch.float)
	X2=torch.tensor(X2,dtype=torch.float)
	adj_augment=torch.tensor(adj_augment,dtype=torch.float)
	adj_pure=torch.tensor(adj_pure,dtype=torch.float)
	print("done")
	best_features=train(adata,X,X2,adj_pure,adj_augment,df_meta,n_domains)
	print(sample)
	_,ARI=cluster(adata,best_features,df_meta,n_domains,refined=True)
	data.plot_domains(adata, data_name)
	adata.write(os.path.join(save_path, f'{data_name}.h5ad'),compression="gzip")
