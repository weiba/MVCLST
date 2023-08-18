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
from img_feature import image_feature

from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from typing import Union, Callable
from PIL import Image
from utils import *
from sklearn import metrics
from augment import augment_adata
from sklearn.metrics import silhouette_score

class run():
    def __init__(
        self,
        save_path="./",
        pre_epochs=1000, 
        epochs=500,
        pca_n_comps = 200,
        linear_encoder_hidden=[32,20],
        linear_decoder_hidden=[32],
        conv_hidden=[32,8],
        verbose=True,
        platform='Visium',
        vit_type='vit_b',	
        p_drop=0.01,
        dec_cluster_n=20,
        n_neighbors=15,
        min_cells=3,
        use_gpu = True,
        ):
        self.save_path = save_path
        self.pre_epochs = pre_epochs
        self.epochs = epochs
        self.pca_n_comps = pca_n_comps
        self.linear_encoder_hidden = linear_encoder_hidden
        self.linear_decoder_hidden = linear_decoder_hidden
        self.conv_hidden = conv_hidden
        self.verbose = verbose
        self.platform = platform
        self.vit_type = vit_type
        self.p_drop = p_drop
        self.dec_cluster_n = dec_cluster_n
        self.n_neighbors = n_neighbors
        self.min_cells = min_cells
        self.platform = platform
        self.use_gpu = use_gpu

    
    def _get_augment(
		self,
		adata,
		adjacent_weight = 0.3,
		neighbour_k = 4,
		weights = "weights_matrix_all",
		spatial_k = 30,
		):
        adata_augment = augment_adata(adata, 
								adjacent_weight = adjacent_weight,
								neighbour_k = neighbour_k,
								platform = self.platform,
								weights = weights,
								spatial_k = spatial_k,
								)
        print("Step 1: Augment gene representation is Done!")
        return adata_augment
    def _get_adata(self,data_path,data_name):
        adata = read_10X_Visium(os.path.join(data_path, data_name))
        save_path_image_crop = Path(os.path.join(self.save_path, 'Image_crop', f'{data_name}'))
        save_path_image_crop.mkdir(parents=True, exist_ok=True)
        adata = image_feature.image_crop(adata, save_path=save_path_image_crop)
        adata = image_feature(adata, pca_components = self.pca_n_comps, vit_type = self.vit_type).extract_image_feat()
        save_data_path = Path(os.path.join(self.save_path, f'{data_name}'))
        save_data_path.mkdir(parents=True, exist_ok=True)
        adata.write(os.path.join(self.save_path, f'{data_name}.h5ad'),compression="gzip")
        return adata
    def plot_domains(self, 
                adata, 
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
            save_path_figure = Path(os.path.join(self.save_path, "Figure", data_name))
            save_path_figure.mkdir(parents=True, exist_ok=True)
            plt.savefig(os.path.join(save_path_figure,f'{data_name}_domains.pdf'), bbox_inches='tight', dpi=dpi)
        else:
            pass
    def image_crop(
            adata,
            save_path,
            library_id=None,
            crop_size=100,
            target_size=224,
            verbose=False,
            ):
            if library_id is None:
                library_id = list(adata.uns["spatial"].keys())[0]

            image = adata.uns["spatial"][library_id]["images"][
                    adata.uns["spatial"][library_id]["use_quality"]]
            if image.dtype == np.float32 or image.dtype == np.float64:
                image = (image * 255).astype(np.uint8)
            img_pillow = Image.fromarray(image)
            tile_names = []

            with tqdm(total=len(adata),
                    desc="Tiling image",
                    bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:
                for imagerow, imagecol in zip(adata.obs["imagerow"], adata.obs["imagecol"]):
                    imagerow_down = imagerow - crop_size / 2
                    imagerow_up = imagerow + crop_size / 2
                    imagecol_left = imagecol - crop_size / 2
                    imagecol_right = imagecol + crop_size / 2
                    tile = img_pillow.crop(
                        (imagecol_left, imagerow_down, imagecol_right, imagerow_up))
                    tile.thumbnail((target_size, target_size), Image.ANTIALIAS) ##### 
                    tile.resize((target_size, target_size)) ###### 
                    tile_name = str(imagecol) + "-" + str(imagerow) + "-" + str(crop_size)
                    out_tile = Path(save_path) / (tile_name + ".png")
                    tile_names.append(str(out_tile))
                    if verbose:
                        print(
                            "generate tile at location ({}, {})".format(
                                str(imagecol), str(imagerow)))
                    tile.save(out_tile, "PNG")
                    pbar.update(1)

            adata.obs["slices_path"] = tile_names
            if verbose:
                print("The slice path of image feature is added to adata.obs['slices_path'] !")
            return adata
