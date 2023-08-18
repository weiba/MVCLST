import os
import time
import torch
import math
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import pairwise_distances
import anndata
from pathlib import Path
from scipy.sparse import issparse, isspmatrix_csr, csr_matrix, spmatrix
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.spatial import distance
from sklearn.decomposition import IncrementalPCA, NMF, TruncatedSVD, FactorAnalysis, LatentDirichletAllocation, sparse_encode
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from tqdm import tqdm

def cal_gene_weight(
	data,
	n_components=50,
	gene_dist_type = "cosine",
	):
	pca = PCA(n_components = n_components)
	if isinstance(data, np.ndarray):
		data_pca = pca.fit_transform(data)
	elif isinstance(data, csr_matrix):
		data = data.toarray()
		data_pca = pca.fit_transform(data)
	gene_correlation = 1 - pairwise_distances(data_pca, metric = gene_dist_type)
	return gene_correlation


def cal_weight_matrix(
		adata,
		platform = "Visium",
		pd_dist_type="euclidean",
		md_dist_type="cosine",
		gb_dist_type="correlation",
		n_components = 50,
		no_morphological = False,
		spatial_k = 30,
		spatial_type = "KDTree",
		verbose = True,
		):
		img_row = adata.obs["imagerow"]
		img_col = adata.obs["imagecol"]
		array_row = adata.obs["array_row"]
		array_col = adata.obs["array_col"]
		rate = 2
		reg_row = LinearRegression().fit(array_row.values.reshape(-1, 1), img_row)
		reg_col = LinearRegression().fit(array_col.values.reshape(-1, 1), img_col)
		physical_distance = pairwise_distances(
									adata.obs[["imagecol", "imagerow"]], 
								  	metric=pd_dist_type)
		unit = math.sqrt(reg_row.coef_ ** 2 + reg_col.coef_ ** 2)
		physical_distance = np.where(physical_distance >= rate * unit, 0, 1)
		physical_distance1 = np.where(physical_distance >= 1.5*unit, 0, 1)

		

		print("Physical distance calculting Done!")
		print("The number of nearest tie neighbors in physical distance is: {}".format(physical_distance.sum()/adata.shape[0]))
	
	########### gene_expression weight
		gene_counts = adata.X.copy()#adata.obsm["pca_data"].copy()
		gene_correlation = cal_gene_weight(data = gene_counts, 
											gene_dist_type = gb_dist_type, 
											n_components = n_components)
	# gene_correlation[gene_correlation < 0 ] = 0
		print("Gene correlation calculting Done!")
		if verbose:
			adata.obsm["gene_correlation"] = gene_correlation
			adata.obsm["physical_distance"] = physical_distance
			adata.obsm["physical_distance1"] = physical_distance1

		morphological_similarity = 1 - pairwise_distances(np.array(adata.obsm["image_feat_pca"]), metric=md_dist_type)
		morphological_similarity[morphological_similarity < 0] = 0
		print("Morphological similarity calculting Done!")
		if verbose:
			adata.obsm["morphological_similarity"] = morphological_similarity	
		adata.obsm["weights_matrix_all"] = (physical_distance
												*gene_correlation
												*morphological_similarity)

		adata.obsm["weights_matrix_nomd"] = (gene_correlation
												*physical_distance)	
		print("The weight result of image feature is added to adata.obsm['weights_matrix_all'] !")
		return adata


def find_adjacent_spot(
	adata,
	use_data = "raw",
	neighbour_k = 4,
	weights='weights_matrix_all',
	verbose = False,
	):
	if use_data == "raw":
		if isinstance(adata.X, csr_matrix):
			gene_matrix = adata.X.toarray()
		elif isinstance(adata.X, np.ndarray):
			gene_matrix = adata.X
		elif isinstance(adata.X, pd.Dataframe):
			gene_matrix = adata.X.values
		else:
			raise ValueError(f"""{type(adata.X)} is not a valid type.""")
	else:
		# sc.pp.filter_genes(adata, min_cells=3)
		# sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
		# adata_X = sc.pp.normalize_total(adata, target_sum=1, exclude_highly_expressed=True, inplace=False)['X']
		# adata_X = sc.pp.log1p(adata_X)
		# adata_X = sc.pp.scale(adata_X)
		# inputs = sc.pp.pca(adata_X, n_comps=300)
		# adata.obsm["pca_data"]=inputs.copy()
		gene_matrix = adata.obsm["pca_data"]
	weights_matrix = adata.obsm[weights]
	weights_list = []
	final_coordinates = []
	with tqdm(total=len(adata), desc="Find adjacent spots of each spot",
                  bar_format="{l_bar}{bar} [ time left: {remaining} ]",) as pbar:
		for i in range(adata.shape[0]):
			if weights == "physical_distance":
				current_spot = adata.obsm[weights][i].argsort()[-(neighbour_k+3):][:(neighbour_k+2)]
			else:
				current_spot = adata.obsm[weights][i].argsort()[-neighbour_k:][:neighbour_k-1]
			spot_weight = adata.obsm[weights][i][current_spot]
			spot_matrix = gene_matrix[current_spot]
			if spot_weight.sum() > 0:
				spot_weight_scaled = (spot_weight / spot_weight.sum())
				weights_list.append(spot_weight_scaled)
				spot_matrix_scaled = np.multiply(spot_weight_scaled.reshape(-1,1), spot_matrix)
				spot_matrix_final = np.sum(spot_matrix_scaled, axis=0)
			else:
				spot_matrix_final = np.zeros(gene_matrix.shape[1])
				weights_list.append(np.zeros(len(current_spot)))
			final_coordinates.append(spot_matrix_final)
			pbar.update(1)
		adata.obsm['adjacent_data'] = np.array(final_coordinates)
		if verbose:
			adata.obsm['adjacent_weight'] = np.array(weights_list)
		return adata


def augment_gene_data(
	adata,
	use_data = "raw",
	adjacent_weight = 0.2,
	):
	adjacent_gene_matrix = adata.obsm["adjacent_data"].astype(float)
	if use_data == "raw":
		if isinstance(adata.X, np.ndarray):
			augement_gene_matrix =  adata.X + adjacent_weight * adjacent_gene_matrix
		elif isinstance(adata.X, csr_matrix):
			augement_gene_matrix = adata.X.toarray() + adjacent_weight * adjacent_gene_matrix
	# adata.obsm["augment_gene_data"] = augement_gene_matrix
	elif use_data == "pca_data":
		if isinstance(adata.obsm["pca_data"], np.ndarray):
			augement_gene_matrix =  adata.obsm["pca_data"] + adjacent_weight * adjacent_gene_matrix
		else:
			augement_gene_matrix = adata.obsm["pca_data"].toarray() + adjacent_weight * adjacent_gene_matrix
	adata.obsm["augment_gene_data"] = augement_gene_matrix
	del adjacent_gene_matrix
	return adata

def augment_adata(
	adata,
	platform = "Visium",
	pd_dist_type="euclidean",
	md_dist_type="cosine",
	gb_dist_type="correlation",
	n_components = 100,
	no_morphological = False,
	use_data = "raw",
	neighbour_k = 4,
	weights = "weights_matrix_all",
	adjacent_weight = 0.2,
	spatial_k = 30,
	spatial_type = "KDTree"
	):
	adata = cal_weight_matrix(
				adata,
				platform = platform,
				pd_dist_type = pd_dist_type,
				md_dist_type = md_dist_type,
				gb_dist_type = gb_dist_type,
				n_components = n_components,
				no_morphological = no_morphological,
				spatial_k = spatial_k,
				spatial_type = spatial_type,
				)
	adata = find_adjacent_spot(adata,
				use_data = use_data,
				neighbour_k = neighbour_k,
				weights = weights)
	adata = augment_gene_data(adata,
			   	use_data = use_data,
				adjacent_weight = adjacent_weight,)
	return adata





