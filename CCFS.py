import os 
import torch.nn as nn
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
from utils_copy import *
from sklearn.metrics import silhouette_score
from augment import *
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics import normalized_mutual_info_score as nmi_score
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 7)
        )

    def forward(self, x):
        return self.layers(x)

def load_data(filename):
    data = pd.read_csv(filename)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return X, y

def train_MLP(model, device, loader, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(loader.dataset)} ({100. * batch_idx / len(loader):.0f}%)]\tLoss: {loss.item():.6f}')

def _compute_smaps(model, loader, device):
    model.eval()
    smap_dict = {}

    for data, target in loader:
        data, target = data.to(device), target.to(device)
        data.requires_grad = True

        output = model(data)
        predicted_classes = output.argmax(dim=1)

        for class_id in torch.unique(predicted_classes):
            class_gradients = []
            class_mask = (predicted_classes == class_id)

            if class_mask.sum() > 0:
                for idx in torch.where(class_mask)[0]:
                    model.zero_grad()
                    class_score = output[idx, class_id]
                    class_score.backward(retain_graph=True)
                    class_gradients.append(data.grad[idx].cpu().numpy())
                    data.grad.data.zero_()

                mean_gradients = np.mean(class_gradients, axis=0)
                smap_dict[class_id.item()] = mean_gradients

    return smap_dict

def computer_smaps():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, y = load_data('gene_expression_data.csv')
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    input_size = X.shape[1]
    model = MLP(input_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_MLP(model, device, loader, optimizer, criterion, epochs=10)

    smap = _compute_smaps(model, loader, device)
    print("Smap (gene influence matrix for each class):")
    for class_id, influence in smap.items():
        print(f"Class {class_id}: {influence}")

sample_list = ['151673']#'151509','151510','151669','151670','151671','151672','151673','151674','151675','151676'
all_labels = []

for sample in sample_list:
    if data_name in ['151669','151670','151671','151672']:
        n_domains = 5
    else:
        n_domains = 7
    for i in range(1,1000):
        data_path = "/home/zhangzhihao/MVCLST/data/151673" #### to your path
        data_name = sample
        save_path = data_path+"/"+sample+"/" #### save path
        save_path_figure = Path(os.path.join(save_path, "Figure", data_name))
        save_path_figure.mkdir(parents=True, exist_ok=True)
        data = run(save_path = save_path, 
            platform = "Visium",
            pca_n_comps = 128,
            pre_epochs = 800,
            vit_type='vit_b',#'vit'
            )
        df_meta = pd.read_csv(data_path+'/'+data_name+'/metadata.tsv', sep='\t')
        df_meta['Ground Truth'] = df_meta['layer_guess_reordered'].astype('category').cat.codes
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
        Leiden_labels = adata.obs['Leiden'].values
        all_labels.append(Leiden_labels)
        data.plot_domains(adata, data_name)
        adata.write(os.path.join(save_path, f'{data_name}.h5ad'),compression="gzip")

    all_mclust_labels_array = np.vstack(all_labels)
    cons_mat = compute_consensus_matrix(all_mclust_labels_array)
    linkage_matrix = hierarchy.linkage(cons_mat, method='average', metric='euclidean')
    consensus_labels = hierarchy.fcluster(linkage_matrix, n_domains, criterion='maxclust')
    consensus_labels = convert_labels(consensus_labels)

ARI = ari_score(df_meta['Ground Truth'], consensus_labels)
NMI = nmi_score(df_meta['Ground Truth'], consensus_labels)