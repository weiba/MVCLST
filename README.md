README
===============================
MVCLST:A multi-view comparative learning method for spatial transcriptomics data clustering 
This document mainly introduces the python code of MVCLST algorithm.

# Requirements
- pytorch==1.12.1
- numpy==1.21.0
- scipy==1.10.1 
- scikit-learn=0.24.1
- torch-geometric==2.3.0
- torch-sparse ==0.6.16
- scanpy==1.7.2
- python==3.8.16
- h5py==2.10.0 
- anndata==0.7.6

# Instructions
This project includes all the codes for the MVCLST algorithm experimented on the dataset (DLPFC). We only introduce the algorithm proposed in our paper, MVCLST, and the introduction of other algorithms can be found in the corresponding paper.

# Model composition and meaning
NIHGCN is composed of common modules and experimental modules.

## Common module
- Data defines the data used by the model
	- DLPFC
		- 151673
			- filtered_feature_bc_matrix.h5
			- metadata.tsv
			- spatial
				- histology.tif
				- scalefactors_json.json
				- tissue_hires_image.png
				- tissue_lowres_image.png
				- tissue_positions_list.csv
- augment.py defines the augmentation of the model			
- get_data.py defines the data loading of the model.
- MVCLST.py defines the complete MVCLST model.
- utils.py and untils_copy.py defines the tool functions needed by the entire algorithm during its operation.
- img_feature.py defines the method to extract image feature of the model.

## Experimental module
 main.py files are capable of conducting all data experiments within the same dataset. In subsequent statistical analyses, we examine the output of the main files. The utils.py file encompasses all tools necessary for the performance and analysis of the entire experiment, including calculations for ARI, NMI scores, and data transformations. All functions are developed using PyTorch and support CUDA.

# Contact
If you have any question regard our code or data, please do not hesitate to open a issue or directly contact me (weipeng1980@gmail.com).
