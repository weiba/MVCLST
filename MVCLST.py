from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import scanpy as sc
from scipy.spatial import distance
from sklearn import metrics
import pandas as pd
from utils import *
import torch_geometric.nn as gnn
from torch_geometric.nn import GCNConv
import torch.autograd as autograd
import random
from sklearn.decomposition import PCA

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        x = torch.matmul(x, self.weight)
        x = torch.matmul(adj, x) + self.bias
        return x
class VAE(nn.Module):
    def __init__(self, input_dim, input_dim1, hidden_dim,hidden_dim1):
        super(VAE, self).__init__()
        self.encoder_vae = nn.Sequential(nn.Linear(input_dim, hidden_dim))
        self.fc_mu = nn.Linear(hidden_dim, hidden_dim)
        self.fc_logvar = nn.Linear(hidden_dim, hidden_dim)
        self.gene_encoder=GraphConvolution(input_dim, hidden_dim1)
        self.decoder = nn.Sequential(nn.Linear(hidden_dim+hidden_dim1, input_dim))
        self.encoder_gcn = GraphConvolution(input_dim,hidden_dim1)
        self.encoder_gcn1 =GraphConvolution(input_dim,hidden_dim1)
        self.decoder_gcn1 = GraphConvolution(2*hidden_dim1,input_dim)
        self.mse = nn.MSELoss()
        self.batchnorm = nn.BatchNorm1d(hidden_dim)
        self.batchnorm2 = nn.BatchNorm1d(hidden_dim1)
        self.prelu=nn.ReLU(hidden_dim1)
    def encode(self, x):
        encoded = self.encoder_vae(x)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        decoded = self.decoder(z)
        return decoded

    def forward(self,x,x2,adj_aug,adj_pure):
            #####VAE
            vae_mu, vae_logvar = self.encode(x2)
            vae_z = self.reparameterize(vae_mu, vae_logvar)
            vae_z=self.batchnorm(vae_z)
            gcn_z=self.gene_encoder(x2,adj_pure)
            gcn_z=self.batchnorm2(gcn_z)
            gcn_z=self.no_nan(gcn_z)
            gcn_z=self.prelu(gcn_z)
            vae_rec = self.decode(torch.cat((vae_z,gcn_z), dim=1))
            h = self.encoder_gcn(x,adj_aug)
            h=self.batchnorm2(h)
            h=self.prelu(h)

            h_private = self.encoder_gcn1(x,adj_aug)
            h_private=self.prelu(h_private)
            h1_rec = self.decoder_gcn1(torch.cat((h,h_private), dim=1),adj_aug)
            return vae_z, vae_rec,gcn_z,h, h_private, h1_rec

    def orthogonal_loss(self, features1, features2):
        covariance = torch.matmul(features1, features2.t())
        loss = torch.sum(torch.abs(covariance * (1 - torch.eye(features1.size(0)).to(features1.device))))
        return loss

    def adjacency_loss(self, features, adjacency_matrix, n):
        similarity_matrix = torch.matmul(features, features.t())
        adjacency_pred = torch.mul(similarity_matrix,adjacency_matrix)
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(adjacency_pred, adjacency_matrix.float())

        return loss

    def similarity(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())
    def zscore_normalization(self,data):
        mean = torch.mean(data, dim=0)
        std = torch.std(data, dim=0)
        std[std == 0] = 1.0
        normalized_data = (data - mean) / std

        return normalized_data
    def contrastive_loss(self, mat1, mat2):
        mat1=torch.relu(mat1)
        mat2=torch.relu(mat2)
        emb_similarity = self.similarity(mat1, mat2)
        eye_matrix = torch.eye(emb_similarity.shape[0])
        eye_matrix=eye_matrix.to('cuda')
        numerator = emb_similarity.mul(eye_matrix)
        numerator = numerator.sum(dim=-1) + 1e-8
        denominator = torch.sum(emb_similarity, dim=-1) + 1e-8
        loss = -torch.log(numerator / denominator).mean()

        return loss
    def no_nan(self,X):
        nan_mask = torch.isnan(X)
        x_no_nan = torch.where(nan_mask, torch.zeros_like(X), X)
        return x_no_nan
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x)
def compute_gradient_penalty(critic, real_samples, fake_samples, device):
    """Calculates the gradient penalty loss for WGAN GP"""
    alpha = torch.rand((real_samples.shape[0], 1)).to(device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    critic_interpolates = critic(interpolates)
    fakes = torch.ones((real_samples.shape[0], 1)).to(device)
    gradients = autograd.grad(
        outputs=critic_interpolates,
        inputs=interpolates,
        grad_outputs=fakes,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
def train(adata,X,X2,adj_pure,adj_aug, df_meta,n_domains):
    input_dim = X.shape[1]
    input_dim1 = X.shape[1]
    hidden_dim = 32
    hidden_dim1=32
    model = VAE(input_dim, input_dim1, hidden_dim,hidden_dim1)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    X = X.to(device)
    X2 = X2.to(device)
    adj_aug=adj_aug.to(device)
    adj_pure=adj_pure.to(device)
    Discriminator1=Discriminator(hidden_dim1).to(device)
    d_optimizer = torch.optim.RMSprop(Discriminator1.parameters(), lr=0.002)
    mse_loss=nn.MSELoss()
    best_ari = 0.0
    best_features = None

    epochs =5000
    for epoch in range(epochs):
        d_optimizer.zero_grad()
        optimizer.zero_grad()
        vae_z, vae_rec,gcn_z,h, h_private, h1_rec= model(X,X2,adj_aug,adj_pure)
        d_loss = torch.mean(h)-torch.mean(gcn_z)
        g_p = compute_gradient_penalty(
        critic=Discriminator1,
        real_samples=gcn_z,
        fake_samples=h,
        device=device
        )
        d_loss = d_loss + 10*g_p
        d_loss.backward(retain_graph=True)
        d_optimizer.step()
        loss_vae_rec = mse_loss(vae_rec, X2)
        loss_gcn_rec=mse_loss(h1_rec, X)
        loss_rec = loss_vae_rec  +loss_gcn_rec
        
        loss_graph_rec1 = model.adjacency_loss(vae_rec, adj_aug, n=4)
        loss_graph_rec2 = model.adjacency_loss(h1_rec, adj_pure, n=4)
        loss_gr = loss_graph_rec1 + loss_graph_rec2
        
        loss_con = model.contrastive_loss(gcn_z, h)
        g_loss = -torch.mean(h)
        total_loss =  loss_rec+loss_gr+loss_con+g_loss
        total_loss.backward(retain_graph=True)
        optimizer.step()


        if epoch % 10 == 0:
            print("Epoch [{:}/{:}], Loss: {:.4f}".format(epoch, epochs, total_loss.item()))
            print("Loss_rec: {:.4f},g_loss{:.4f},gr_loss{:.4f},con_loss{:.4f}".format(loss_rec, g_loss, loss_gr,loss_con))
        if epoch >800:
            if epoch % 100 ==0:
                concatenated_features = torch.cat((vae_z,h,h_private), dim=1)
                concatenated_features=model.zscore_normalization(concatenated_features)
                z = concatenated_features.cpu()
                z=z.detach().numpy()
                z= sc.pp.scale(z)
                _,ARI=cluster(adata,z,df_meta,n_domains)
                if ARI > best_ari:
                    best_ari = ARI
                    best_features = z
    
    return best_features
