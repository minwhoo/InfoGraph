import argparse
from pathlib import Path

import torch
import numpy as np
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import GINConv, global_add_pool
from evaluate_embedding import evaluate_embedding

from model import GlobalLocalDiscriminator, FeedForwardNetwork

DATA_DIR = Path("./data/")


class GINEncoder(nn.Module):
    """Graph/patch representation encoder using GIN as base GNN layer."""
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            self.convs.append(
                GINConv(
                    nn.Sequential(
                        nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim)
                    ),
                )
            )
            self.bns.append(
                nn.BatchNorm1d(hidden_dim)
            )

    def forward(self, graph_data):
        x = graph_data.x
        xs = []
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(conv(x, graph_data.edge_index))
            x = bn(x)
            xs.append(x)

        x_patches = torch.cat(xs, dim=1)
        x_global = global_add_pool(x_patches, graph_data.batch)
        return x_global, x_patches


class InfoGraph(nn.Module):
    """Unsupervised graph representation learning framewark"""
    def __init__(self, input_dim, encoder_hidden_dim, num_encoder_layers):
        super().__init__()
        self.gnn_encoder = GINEncoder(input_dim, encoder_hidden_dim, num_encoder_layers)
        embed_dim = encoder_hidden_dim * num_encoder_layers
        self.discriminator = GlobalLocalDiscriminator(
            FeedForwardNetwork(embed_dim),
            FeedForwardNetwork(embed_dim),
        )

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)

    def forward(self, graph_data):
        h_global, h_patches = self.gnn_encoder(graph_data)
        loss = self.discriminator(h_global, h_patches, graph_data.batch)
        return loss

    def encode_graph(self, graph_data):
        """Get graph representation for downstream tasks"""
        h_global, _ = self.gnn_encoder(graph_data)
        return h_global


def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for data in dataloader:
        if data.x is None:
            data.x = torch.ones(data.batch.shape[0])
        data = data.to(device)
        optimizer.zero_grad()
        loss = model(data)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(dataloader.dataset)


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in dataloader:
            if data.x is None:
                data.x = torch.ones(data.batch.shape[0])
            data = data.to(device)
            loss = model(data)
            total_loss += loss.item() * data.num_graphs
    print(total_loss / len(dataloader.dataset))


def evaluate_downstream(model, dataloader, device):
    model.eval()
    total_loss = 0
    x_embeds = []
    ys = []
    with torch.no_grad():
        for data in dataloader:
            ys.append(data.y.cpu().detach().numpy())
            if data.x is None:
                data.x = torch.ones(data.batch.shape[0])
            data = data.to(device)
            x_embed = model.encode_graph(data)
            x_embeds.append(x_embed.cpu().detach().numpy())
    evaluate_embedding(np.vstack(x_embeds), np.concatenate(ys))


def main():
    np.random.seed(0)
    torch.manual_seed(0)
    # --------------------- PARSE ARGS -----------------------
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", choices=['MUTAG', 'PTC_MR', 'REDDIT-BINARY', 'REDDIT-MULTI-5K', 'IMDB-BINARY', 'IMDB-MULTI'], default='MUTAG')
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--num-epoch", type=int, default=20)
    parser.add_argument("--encoder-hidden-dim", type=int, default=32)
    parser.add_argument("--num-encoder-layers", type=int, default=4)

    args = parser.parse_args()

    print("- Args ----------------------")
    for k, v in vars(args).items():
        print(" - {}={}".format(k, v))
    print("-----------------------------")

    # --------------------- LOAD DATASET ---------------------
    print("Loading dataset...")
    dataset = TUDataset(DATA_DIR / args.dataset, name=args.dataset)
    try:
        dataset_num_features = dataset.num_features
        if dataset_num_features == 0:
            dataset_num_features = 1
    except:
        dataset_num_features = 1

    dataloader = DataLoader(dataset, args.batch_size, shuffle=True)

    # --------------------- TRAIN MODEL ----------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = InfoGraph(dataset_num_features, args.encoder_hidden_dim, args.num_encoder_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    evaluate_downstream(model, dataloader, device)
    print("Starting training...")

    for epoch in range(1, args.num_epoch+1):
        train_loss = train(model, dataloader, optimizer, device)
        print("| Epoch: {:3} | Unsupervised Loss: {:10.4f} |".format(epoch, train_loss))

    print("Training finished!")
    evaluate_downstream(model, dataloader, device)


if __name__ == "__main__":
    main()
